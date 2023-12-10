import numpy as np
from scipy.fftpack import dct, idct
from joblib import Parallel, delayed


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


def process_block(i, j, iRange, blockSize, widthV, heightV, curr_f, ref_f, QP):
    min_mae = 123123
    min_axy = 123123
    currentBlock = curr_f[i][j]
    tempMV = None
    # Initialize tempMatch to a zero array of the same shape as currentBlock
    tempMatch = np.zeros_like(currentBlock)

    for ry in range(iRange, -iRange - 1, -1):
        for rx in range(iRange, -iRange - 1, -1):
            ref_x = j * blockSize + rx
            ref_y = i * blockSize + ry
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
    # resE = entropy_encoder(quantB)

    return tempMV, quantB, recons_block


def process_col(i, iRange, blockSize, widthV, heightV, curr_f, ref_f, QP):
    blockNumInWidth = len(curr_f[0])
    blockNumInHeight = len(curr_f)
    mvR = []
    qtcR = []
    reconsR = []
    for x in range(blockNumInWidth):
        mvT, qtcT, reconsT = process_block(i, x, iRange, blockSize, widthV, heightV, curr_f, ref_f, QP)
        mvR.append(mvT)
        qtcR.append(qtcT)
        reconsR.append(reconsT)
    return mvR, qtcR, reconsR


def Full_searchparaT2(currFnum, iRange, currF, refF, blockSize, QP=6, heightV=288, widthV=352):
    blockNumInWidth = len(currF[0])
    blockNumInHeight = len(currF)

    print("The", currFnum, "frame, P")
    blackframe = np.full((heightV, widthV), 128)

    motion_V = []
    QTC_F = []
    reconstructed_frame = []

    curr_f = currF
    ref_frames = refF
    process_block_calls = []
    for i in range(blockNumInHeight):
        # motion_V.append([])
        # QTC_F.append([])
        # reconstructed_frame.append([])

        call = delayed(process_col)(i, iRange, blockSize, widthV, heightV, curr_f, ref_frames, QP)
        process_block_calls.append(call)

    parallel = Parallel(n_jobs=2, prefer="threads")
    results = parallel(process_block_calls)

    for result in results:
        tempMV, quantB, recons_block = result
        motion_V.append(tempMV)
        QTC_F.append(quantB)
        reconstructed_frame.append(recons_block)

    return motion_V, QTC_F, reconstructed_frame
