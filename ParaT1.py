import numpy as np
from scipy.fftpack import dct, idct
from joblib import Parallel, delayed


def reorder(matrix):
    reordered = []
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows + cols - 1):
        if i % 2 == 0:  # Even diagonals
            row = min(i, rows - 1)
            col = max(0, i - rows + 1)
            while row >= 0 and col < cols:
                reordered.append(matrix[row][col])
                row -= 1
                col += 1
        else:  # Odd diagonals
            col = min(i, cols - 1)
            row = max(0, i - cols + 1)
            while col >= 0 and row < rows:
                reordered.append(matrix[row][col])
                col -= 1
                row += 1
    return reordered

def entropy_encoder(matrix):
    reorderM = reorder(matrix)
    res = rle_encode(reorderM)
    return res

def rle_encode(arr):
    encoded = []
    i = 0

    while i < len(arr):
        if arr[i] != 0:  # Non-zero terms
            count_non_zero = 0
            while i < len(arr) and arr[i] != 0:
                count_non_zero += 1
                i += 1
            encoded.extend([-count_non_zero, *arr[i - count_non_zero: i]])
        else:  # Zero terms
            count_zero = 0
            while i < len(arr) and arr[i] == 0:
                count_zero += 1
                i += 1
            encoded.append(count_zero)

    encoded.append(0)  # Signaling end of non-zero terms
    return encoded
def constractFrame(blocked):
    ind = 0
    res = np.array([])
    for y in blocked:
        row = np.array([])
        for x in y:
            ind += 1
            if len(row) == 0:
                row = x
            else:
                row = np.hstack((row, x))
        if len(res) == 0:
            res = row
        else:
            res = np.vstack((res, row))
    res = np.clip(res, 0, 255)
    res = np.array(res, dtype=np.uint8)
    # cv2.imshow('hello', res)
    # cv2.waitKey(40)  # 按 'q' 键退出
    return res

def dct_2d(block):
    # Apply 2D DCT to the input block
    dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    # dct_block = np.round(dct_block)
    return dct_block


def idct_2d(dct_block):
    # Apply 2D inverse DCT to the DCT coefficients
    idct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
    # round to the nearest integer不确定是否需要
    # idct_block = np.round(idct_block)
    # res = np.clip(idct_block, 0, 255)
    return idct_block


def quantization(TC, QP):
    i = TC.shape[0]
    Q = np.zeros((i, i))
    for y in range(i):
        for x in range(i):
            if (x + y) < (i - 1):
                Q[y][x] = 2 ** QP
            elif (x + y) == (i - 1):
                Q[y][x] = 2 ** (QP + 1)
            else:
                Q[y][x] = 2 ** (QP + 2)

    QTC = np.round(TC / Q)
    return QTC


def racelling(QTC, QP):
    i = QTC.shape[0]
    Q = np.zeros((i, i))
    for y in range(i):
        for x in range(i):
            if (x + y) < (i - 1):
                Q[y][x] = 2 ** QP
            elif (x + y) == (i - 1):
                Q[y][x] = 2 ** (QP + 1)
            else:
                Q[y][x] = 2 ** (QP + 2)

    res = np.round(QTC * Q)
    return res


def MAE(block_A=np.array([[]]), block_B=np.array([[]])):
    """
    Args:
        block_A (2D
        block_B (2D
    """
    absD = np.abs(block_A - block_B)
    mae = absD.mean()
    return mae


def process_block(i, j, iRange, blockSize, widthV, heightV, curr_f, QP):
    min_mae = 123123
    min_axy = 123123
    currentBlock = curr_f[i][j]
    tempMV = None
    # Initialize tempMatch to a zero array of the same shape as currentBlock
    tempMatch = np.zeros_like(currentBlock)
    blackframe = np.full((heightV,widthV), 128)
    ref_f = blackframe

    for ry in range(iRange, -iRange - 1, -1):
        for rx in range(iRange, -iRange - 1, -1):
            ref_x = j * blockSize + rx
            ref_y = i * blockSize + ry
            # print(blockSize)
            if (
                    0 <= ref_x + blockSize <= widthV
                    and 0 <= ref_y + blockSize <= heightV
            ) and (
                    0 <= ref_x <= widthV
                    and 0 <= ref_y <= heightV
            ):
                refBlock = ref_f[ref_y:ref_y + blockSize, ref_x:ref_x + blockSize]
                maeT = MAE(currentBlock, refBlock)
                axy = np.abs(ry) + np.abs(rx)

                if maeT <= min_mae and axy <= min_axy:
                    min_mae = maeT
                    min_axy = axy
                    tempMatch = refBlock
                    tempMV = (rx, ry)

    residual_block = currentBlock - tempMatch

    transed = dct_2d(residual_block)
    quantB = quantization(transed, QP)
    racB = racelling(quantB, QP)
    rtB = idct_2d(racB)
    recons_block = tempMatch + rtB
    #
    resE = entropy_encoder(quantB)

    return tempMV, resE, recons_block


def searchT1(currFnum, iRange, currF, blockSize, QP=6, heightV=288, widthV=352):
    blockNumInWidth = len(currF[0])
    blockNumInHeight = len(currF)

    print("The", currFnum, "frame, P")
    blackframe = np.full((heightV, widthV), 128)

    motion_V = []
    QTC_F = []
    reconstructed_frame = []

    curr_f = currF


    for i in range(blockNumInHeight):
        motion_V.append([])
        QTC_F.append([])
        reconstructed_frame.append([])

        process_block_calls = []
        for j in range(blockNumInWidth):
            call = delayed(process_block)(i, j, iRange, blockSize, widthV, heightV, curr_f, QP)
            process_block_calls.append(call)

        parallel = Parallel(n_jobs=1, prefer="threads")
        results = parallel(process_block_calls)

        for result in results:
            tempMV, quantB, recons_block = result
            motion_V[i].append(tempMV)
            QTC_F[i].append(quantB)
            reconstructed_frame[i].append(recons_block)

    r_frame = constractFrame(reconstructed_frame)
    return motion_V, QTC_F, r_frame
