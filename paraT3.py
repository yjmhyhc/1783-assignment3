import numpy as np
from scipy.fftpack import dct, idct
from joblib import Parallel, delayed
import threading
import time
import concurrent.futures

frame_lock = threading.Lock()
encodedFrame = np.array([[1, 2, 3], [4, 5, 6]])


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


def process_block(i, j, iRange, blockSize, widthV, heightV, curr_f, ref_f, QP):
    min_mae = float('inf')
    min_axy = float('inf')
    currentBlock = curr_f[i][j]
    tempMV = None
    tempMatch = None

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


def initialize_events(n):
    # 创建一个Event对象列表
    return [threading.Event() for _ in range(n)]


def Full_search_First(currFnum, iRange, currF, refF, blockSize, start_events, encodedF, QP=6, heightV=288,
                      widthV=352):
    blockNumInWidth = len(currF[0])
    blockNumInHeight = len(currF)
    global encodedFrame

    print("The", currFnum, "frame, P")
    blackframe = np.full((heightV, widthV), 128)

    # Initialize lists to store motion vectors and residual data for each block
    motion_V = []
    QTC_F = []
    reconstructed_frame = []

    # Get the current and reference frames
    curr_f = currF
    ref_frames = refF

    for j in range(blockNumInHeight):
        tempMV, quantB, recons_block = process_col(j, iRange, blockSize, widthV, heightV, curr_f, ref_frames, QP)

        motion_V.append(tempMV)
        QTC_F.append(quantB)
        reconstructed_frame.append(recons_block)
        rT = constractFrame(reconstructed_frame)
        with frame_lock:
            encodedFrame = np.empty_like(rT)
            encodedFrame[:] = rT
        if j > 1.9:
            start_events[j - 2].set()
            # start_events[j-2].clear()
    start_events[blockNumInHeight - 2].set()
    start_events[blockNumInHeight - 1].set()
    # start_events[blockNumInHeight-1].clear()
    # start_events[blockNumInHeight - 2].clear()


    return motion_V, QTC_F, reconstructed_frame


def Full_search_Second(currFnum, iRange, currF, refF, blockSize, start_events, QP=6, heightV=288, widthV=352):
    global encodedFrame
    blockNumInHeight = len(currF)

    # print("The", currFnum, "frame, P")

    # Initialize lists to store motion vectors and residual data for each block
    motion_V = []
    QTC_F = []
    reconstructed_frame = []

    # Get the current and reference frames
    curr_f = currF


    for j in range(blockNumInHeight):
        start_events[j].wait()
        # print("2j", j, "len REF", len(encodedFrame))

        ref_frames = encodedFrame
        tempMV, quantB, recons_block = process_col(j, iRange, blockSize, widthV, heightV, curr_f, ref_frames, QP)
        motion_V.append(tempMV)
        QTC_F.append(quantB)
        reconstructed_frame.append(recons_block)

    return motion_V, QTC_F, reconstructed_frame


def T3main(currFnum, iRange, currF1, currF2, refF, blockSize, QP=6, heightV=288, widthV=352):
    # blockNumInHeight = len(currF1)
    #
    # start_events = initialize_events(blockNumInHeight)
    # global encodedFrame
    # # 创建两个线程
    # boss_thread = threading.Thread(target=Full_search_First, args=(
    #     currFnum, iRange, currF1, refF, blockSize, start_events, encodedFrame, QP, heightV, widthV))
    # worker_thread = threading.Thread(target=Full_search_Second, args=(
    #     currFnum, iRange, currF2, encodedFrame, blockSize, start_events, QP, heightV, widthV))
    #
    # # 启动两个线程
    # boss_thread.start()
    # worker_thread.start()
    #
    # # 等待两个线程完成
    # boss_thread.join()
    # worker_thread.join()
    # for i in start_events:
    #     i.clear()


    start_events = initialize_events(len(currF1))
    global encodedFrame

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the thread functions to the executor
        future_first = executor.submit(Full_search_First, currFnum, iRange, currF1, refF, blockSize, start_events,
                                       encodedFrame, QP, heightV, widthV)
        future_second = executor.submit(Full_search_Second, currFnum, iRange, currF2, encodedFrame, blockSize,
                                        start_events, QP, heightV, widthV)

        # Retrieve the results from the futures
        result_first = future_first.result()
        result_second = future_second.result()

    # Clear the events
    for event in start_events:
        event.clear()

    return result_first, result_second

