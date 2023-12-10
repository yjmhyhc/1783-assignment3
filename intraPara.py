from joblib import Parallel, delayed, memory
import numpy as np
from collections import defaultdict
from Encoder_a3._RCpy_perrow import dct_2d, idct_2d, entropy_encoder, mse, MAE,quantization, racelling


def quantizationN(TC, Q):
    QTC = np.round(TC / Q)
    return QTC


def racellingN(QTC, Q):
    res = np.round(QTC * Q)
    return res


def process_block(i, j, curr_f, blockSize, reconstructed_frame, Q):
    currentBlock = curr_f[i][j]
    blackB = np.full((blockSize, blockSize), 128, np.uint8)
    if j == 0:
        predB0 = np.copy(blackB)
    else:
        predB0 = np.copy(reconstructed_frame[i][j - 1])
    if i == 0:
        predB1 = np.copy(blackB)
    else:
        predB1 = np.copy(reconstructed_frame[i - 1][j])

    predB0[:] = predB0[:, -1][:, np.newaxis]
    predB1[:] = predB1[-1, :][np.newaxis, :]
    mae0 = np.mean(np.abs(currentBlock - predB0))
    mae1 = np.mean(np.abs(currentBlock - predB1))
    mode = 1 if mae0 > mae1 else 0
    predB = predB1 if mode else predB0
    residual_block = currentBlock - predB
    transed = dct_2d(residual_block)
    quantB = quantizationN(transed, Q)
    racB = racellingN(quantB, Q)
    rtB = idct_2d(racB)
    recons_block = predB + rtB
    resE = entropy_encoder(quantB)
    return mode, resE, recons_block


def intraperdicationpara(currF, blockSize, QP=6):
    Q = np.zeros((blockSize, blockSize))
    for y in range(blockSize):
        for x in range(blockSize):
            if (x + y) < (blockSize - 1):
                Q[y][x] = 2 ** QP
            elif (x + y) == (blockSize - 1):
                Q[y][x] = 2 ** (QP + 1)
            else:
                Q[y][x] = 2 ** (QP + 2)
    blockNumInWidth = len(currF[0])
    curr_f = currF
    blackB = np.full((blockSize, blockSize), 128, np.uint8)
    reconstructed_frame = [[blackB] * blockNumInWidth for _ in range(len(curr_f))]

    coords = defaultdict(list)
    for i in range(len(curr_f)):
        for j in range(blockNumInWidth):
            coords[i + j].append((i, j))

    mode = []
    approximated_res = []
    for diagonal in range(len(curr_f) + blockNumInWidth - 1):
        results = Parallel(n_jobs=2, prefer="threads")(
            delayed(process_block)(i, j, curr_f, blockSize, reconstructed_frame, Q) for i, j in coords[diagonal])

        for i, result in enumerate(results):
            m, resE, recons_block = result
            x, y = coords[diagonal][i]
            mode.append(m)
            approximated_res.append(resE)
            reconstructed_frame[x][y] = recons_block

    return mode, approximated_res, reconstructed_frame


def process_blockV(currentBlock, block_x, block_y, QP, lam, reconstructed_frame,blockSize):
    # Unvariable block size prediction
    # print("start block:", block_x, block_y)

    blackB = np.full((blockSize, blockSize), 128, np.uint8)
    if block_x == 0:
        predB0 = np.copy(blackB)
    else:
        predB0 = np.copy(reconstructed_frame[block_y][block_x - 1])
    if block_y == 0:
        predB1 = np.copy(blackB)
    else:
        predB1 = np.copy(reconstructed_frame[block_y - 1][block_x])
    for x in range(blockSize):
        for y in range(blockSize):
            predB0[y][x] = predB0[y][-1]
    for x in range(blockSize):
        for y in range(blockSize):
            predB1[y][x] = predB1[-1][x]
    mae0 = mse(currentBlock, predB0)
    mae1 = mse(currentBlock, predB1)
    if mae0 > mae1:
        predB_UV = predB1
        modeUV = 1
    else:
        predB_UV = predB0
        modeUV = 0

    # Variable block size prediction
    residual_block = currentBlock - predB_UV
    transed = dct_2d(residual_block)
    quantB = quantization(transed, QP)
    racB = racelling(quantB, QP)
    rtB = idct_2d(racB)
    recons_block = predB_UV + rtB
    resE = entropy_encoder(quantB)
    size_UV = len(resE)
    QTC_UV = resE

    '''
                    varible size part
                    '''
    ref_b = [[[[], []], [[], []]],
             [[[], []], [[], []]]]
    # for i1 in range(2):
    #     for j1 in range(2):
    #         ref_b[i1].append([[], []])
    # 提取最右一列
    right_column = predB0[:, -1]
    # 获取列的中间索引
    middle_index = len(right_column) // 2
    # print("Luck",middle_index)
    # 将最右一列分成两半
    f_h = right_column[:middle_index]
    second_half = right_column[middle_index:]
    ref_b[0][0][0] = f_h
    ref_b[1][0][0] = second_half

    # 提取最后一行
    last_row = predB1[-1, :]
    middle_index_row = len(last_row) // 2
    # print("LUCK",middle_index_row)
    # 将最后一行分成两半
    first_half_row = last_row[:middle_index_row]
    second_half_row = last_row[middle_index_row:]
    ref_b[0][0][1] = first_half_row
    ref_b[0][1][1] = second_half_row

    middle = middle_index

    Ori_VB = [[0, 0], [0, 0]]
    Ori_VB[0][0] = currentBlock[:middle, :middle]
    Ori_VB[0][1] = currentBlock[:middle, middle:]
    Ori_VB[1][0] = currentBlock[middle:, :middle]
    Ori_VB[1][1] = currentBlock[middle:, middle:]
    size_V = 0
    predBV = [[0, 0], [0, 0]]
    Mode_VB = [[0, 0], [0, 0]]
    Recon_VSB = [[0, 0], [0, 0]]
    QTC_V = [[0, 0], [0, 0]]
    # blank_matrix = np.zeros((middle_index, middle_index))
    for y2 in range(2):
        for x2 in range(2):
            # print("block:", i, j, "sub block", y2, x2)
            preSB0_V = np.zeros((middle_index, middle_index))
            for x in range(middle_index):
                for y in range(middle_index):
                    preSB0_V[y][x] = ref_b[y2][x2][0][y]
            preSB1_V = np.zeros((middle_index, middle_index))
            for x in range(middle_index):
                for y in range(middle_index):
                    # print()
                    preSB1_V[y][x] = ref_b[y2][x2][1][x]
            # tbd
            mae0 = MAE(Ori_VB[y2][x2], preSB0_V)
            mae1 = MAE(Ori_VB[y2][x2], preSB1_V)
            if mae0 > mae1:
                predBV[y2][x2] = preSB1_V
                Mode_VB[y2][x2] = 1
            else:
                predBV[y2][x2] = preSB0_V
                Mode_VB[y2][x2] = 0

            residual_Sb = Ori_VB[y2][x2] - predBV[y2][x2]

            transedSB = dct_2d(residual_Sb)
            quantSB = quantizationN(transedSB, QP - 1)
            racSB = racellingN(quantSB, QP - 1)
            rtSB = idct_2d(racSB)
            recons_sblock = predBV[y2][x2] + rtSB
            resSubBlockEntropy = entropy_encoder(quantSB)
            QTC_V[y2][x2] = resSubBlockEntropy
            size_V += len(resSubBlockEntropy)

            # print(resSubBlockEntropy)
            Recon_VSB[y2][x2] = recons_sblock
            if y2 + 1 < 2:
                ref_b[y2 + 1][x2][1] = recons_sblock[-1, :]
                # # 提取最后一行
            if x2 + 1 < 2:
                ref_b[y2][x2 + 1][0] = recons_sblock[:, -1]

    result_upper = np.concatenate((Recon_VSB[0][0], Recon_VSB[0][1]), axis=1)
    result_lower = np.concatenate((Recon_VSB[1][0], Recon_VSB[1][1]), axis=1)
    resR_sb = np.concatenate((result_upper, result_lower), axis=0)

    D_UV = mse(currentBlock, recons_block)
    D_V = mse(currentBlock, resR_sb)
    J_UV = D_UV + lam * size_UV
    J_V = D_V + lam * size_V
    if J_UV > J_V:
        VaribleBlockFlag = 1
        QTC = QTC_V
        # reconstructed_frame[i].append(resR_sb)
        resB = resR_sb
        mode = Mode_VB
    else:
        VaribleBlockFlag = 0
        QTC = QTC_UV
        # reconstructed_frame[i].append(recons_block) # todo add them in the after
        resB = recons_block
        mode = modeUV
    # print("end block:", block_x, block_y)
    return VaribleBlockFlag, QTC, resB, mode


def intra_Pred_T2(curr_f, currFnum, blocksize, QP=6):

    print("The", currFnum, "frame, I")
    lam = 4 * 2 ** ((QP - 12) / 3)
    blockNumInWidth = len(curr_f[0])
    blockNumInHeight = len(curr_f)
    QTC_F = [[0 for _ in range(blockNumInWidth)] for _ in range(blockNumInHeight)]
    reconstructed_frame = [[0 for _ in range(blockNumInWidth)] for _ in range(blockNumInHeight)]
    mode = [[0 for _ in range(blockNumInWidth)] for _ in range(blockNumInHeight)]
    VaribleBlockFlag = [[0 for _ in range(blockNumInWidth)] for _ in range(blockNumInHeight)]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    coords = defaultdict(list)
    for i in range(len(curr_f)):
        for j in range(blockNumInWidth):
            coords[i + j].append((i, j))

    for diagonal in range(len(curr_f) + blockNumInWidth - 1):
        results = Parallel(n_jobs=1, prefer="threads")(
            delayed(process_blockV)(curr_f[i][j], j, i, QP, lam, reconstructed_frame,blocksize) for i, j in coords[diagonal])

        for i, result in enumerate(results):
            VaribleBlockFlagT, QTC, resB, modeT = result
            x, y = coords[diagonal][i]
            mode[x][y] = modeT
            QTC_F[x][y] = QTC
            reconstructed_frame[x][y] = resB
            VaribleBlockFlag[x][y] = VaribleBlockFlagT

    return mode, QTC_F, reconstructed_frame, VaribleBlockFlag


    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # for i in range(blockNumInHeight):
    #     for j in range(blockNumInWidth):
    #         currentBlock = curr_f[i][j]
    #         VaribleBlockF, QTC, resB, modeT = process_blockV(currentBlock, j, i, QP, lam, reconstructed_frame,blocksize)
    #         VaribleBlockFlag[i][j] = VaribleBlockF
    #         QTC_F[i][j] = QTC
    #         reconstructed_frame[i][j]=(resB)
    #         mode[i][j] = modeT
    # self.VaribleBlockIndicators.append(VaribleBlockFlag)
    # res = self.constractFrame(reconstructed_frame)
    # self.reconstructedFrame[currFnum] = res
    # return mode, QTC_F, reconstructed_frame, VaribleBlockFlag
