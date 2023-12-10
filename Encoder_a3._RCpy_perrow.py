import math
from scipy.fftpack import dct, idct
import numpy as np
import sys
import os
import time
import sys
from intraPara import *
from interPara import Full_searchparaT2
from ParaT1 import searchT1
from paraT3 import T3main
# import pickle as pkl

import cv2


class VideoEncoder:
    def __init__(self, filename, subSampleR, I_p, width=352, height=288):
        '''
        Constructor, initializes the object
        Parameters:
            filename: Input YUV file name
            subSampleR: Subsampling rate, 0->4:4:4, 1->4:2:2, 2->4:2:0
            width: Frame width, default is 288
            height: Frame height, default is 352
        '''
        self.heightV = height
        self.widthV = width
        self.uv_height = height // 2
        self.uv_width = width // 2

        # Calculate frame size based on subsampling
        temp_d = {0: 3, 1: 2, 2: 1.5}
        self.frameSize = int(width * height * (temp_d[int(subSampleR)]))
        print("frames size: ", self.frameSize)

        # Read YUV data from a file and store it in a NumPy array
        self.yuv_data = np.fromfile(filename, dtype=np.uint8, count=-1)
        self.yuv_data = self.yuv_data.astype('int')
        print("total len: ", len(self.yuv_data))
        self.frameNum = len(self.yuv_data) // self.frameSize
        self.yFrame = []
        self.uFrame = []

        self.vFrame = []
        self.frameSpiliting()
        self.iPer = []  # "IPPPP"
        self.I_p = I_p
        for i in range(self.frameNum):
            if i % len(I_p) == 0:
                self.iPer.append(1)
            else:
                self.iPer.append(0)
        # print(self.iPer)

    def frameSpiliting(self):

        for i in range(self.frameNum):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print("this is the", str(i + 1) + "th frame")
            frame_start = i * (self.frameSize)
            y_data = self.yuv_data[frame_start:frame_start +
                                               self.heightV * self.widthV].reshape((self.heightV, self.widthV))
            self.yFrame.append(y_data)
            u_data = self.yuv_data[int(frame_start + self.heightV * self.widthV):int(
                frame_start + 1.25 * self.heightV * self.widthV)].reshape((self.uv_width, self.uv_height))
            self.uFrame.append(u_data)
            v_data = self.yuv_data[int(frame_start + 1.25 * self.heightV * self.widthV):int(
                frame_start + 1.5 * self.heightV * self.widthV)].reshape((self.uv_width, self.uv_height))
            self.vFrame.append(v_data)

        return

    def dispalyY(self):
        for i in range(self.frameNum):
            print("this is the", str(i + 1) + "th frame")
            frame_start = i * (self.frameSize)

            y_data = self.yuv_data[frame_start:frame_start +
                                               self.heightV * self.widthV].reshape((self.heightV, self.widthV))
            cv2.imshow('Y values', y_data)
            if cv2.waitKey(80) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

        return

    def combineY(self):
        for i in range(self.frameNum):
            print("this is the", str(i + 1) + "th frame")
            frame_start = i * (self.frameSize)
            y_data = self.yuv_data[frame_start:frame_start +
                                               self.heightV * self.widthV]
            my_array = np.hstack((my_array, y_data))

        my_array = my_array.astype(np.uint8)
        return (my_array)
        return None

    def blockSpliting(self, blockSize):
        """
        _summary_

        Args:
            blockSize
        """

        blockSize = int(blockSize)
        self.blockSize = blockSize
        self.blockedYF = []

        self.blockNumInWidth = math.ceil(self.widthV / blockSize)
        self.blockNumInHeight = math.ceil(self.heightV / blockSize)

        remainderInWidth = self.widthV % blockSize
        remainderInHeight = self.heightV % blockSize
        paddingInWidth = blockSize - remainderInWidth
        paddingInHeight = blockSize - remainderInHeight

        for fr in range(self.frameNum):
            frame = self.yFrame[fr]
            res = []

            if remainderInWidth != 0:
                grayBlock = np.full(
                    (self.heightV, paddingInWidth), 128, np.uint8)
                frame = np.hstack((frame, grayBlock))
            if remainderInHeight != 0:
                grayBlock = np.full(
                    (paddingInHeight, frame.shape[1]), 128, np.uint8)
                frame = np.vstack((frame, grayBlock))

            # padding and then splitting
            for i in range(self.blockNumInHeight):
                res.append([])
            for i1 in range(self.blockNumInHeight):
                x_start = i1 * blockSize
                x_end = (i1 + 1) * blockSize
                for i2 in range(self.blockNumInWidth):
                    y_start = i2 * blockSize
                    y_end = (i2 + 1) * blockSize
                    block = frame[x_start:x_end, y_start:y_end]
                    res[i1].append(block)

            self.blockedYF.append(res)

        return None

    def MAE(self, block_A=np.array([[]]), block_B=np.array([[]])):
        """

        Args:
            block_A (2D
            block_B (2D
        """
        absD = np.abs(block_A - block_B)
        mae = absD.mean()
        return mae

    def Get_ref_block(self, rx, ry):
        return

    def approximated_residual(self, residual_block, n):
        # residual_block = residual_block % 256
        # return np.round(residual_block / (2 ** n)).astype(int) * (2 ** n)
        return np.round(residual_block / (2 ** n)) * (2 ** n)

    def Full_search(self, currFnum, iRange, QP=6, nRefFrames=1):
        """
        _summary_

        Args:
            currFnum (int): Current frame number
            iRange (int): Search range for motion estimation
            QP (int): Quantization parameter
            nRefFrames (int): Number of reference frames
        """
        print("The", currFnum, "frame, P")
        blackframe = np.full((self.heightV, self.widthV), 128)

        # Initialize lists to store motion vectors and residual data for each block
        motion_V = []
        QTC_F = []
        reconstructed_frame = []

        # Get the current and reference frames
        curr_f = self.blockedYF[currFnum]
        ref_frames = []
        for i in range(nRefFrames):
            if currFnum - i > 0:
                ref_frames.append(self.reconstructedFrame[currFnum - i - 1])
            else:
                ref_frames.append(blackframe)
        # print(ref_frames)

        for i in range(self.blockNumInHeight):
            motion_V.append([])
            QTC_F.append([])
            reconstructed_frame.append([])

            for j in range(self.blockNumInWidth):
                min_mae = float('inf')
                min_axy = float('inf')
                currentBlock = curr_f[i][j]
                tempMV = None
                tempMatch = None
                x = j * self.blockSize
                y = i * self.blockSize
                for ref_frame_idx, ref_f in enumerate(ref_frames):
                    # print(ref_f)
                    if self.FMEEnable == 1:
                        if currFnum - ref_frame_idx > 0:
                            ref_f = self.reconstructedFrame[currFnum - ref_frame_idx - 1]
                        else:
                            ref_f = blackframe
                        for ry in range(iRange, -iRange - 1, -1):
                            for rx in range(iRange, -iRange - 1, -1):
                                ref_x = x + rx
                                ref_y = y + ry
                                # print("refxy: ", ref_x, ref_y)

                                if (
                                        0 <= ref_x + self.blockSize <= self.widthV
                                        and 0 <= ref_y + self.blockSize <= self.heightV
                                ) and (
                                        0 <= ref_x <= self.widthV - 1
                                        and 0 <= ref_y <= self.heightV - 1
                                ):
                                    refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                    # print(ref_f )
                                    maeT = self.MAE(currentBlock, refBlock)
                                    axy = np.abs(ry) + np.abs(rx)

                                    if maeT <= min_mae and axy <= min_axy:
                                        min_mae = maeT
                                        min_axy = axy
                                        tempMatch = refBlock
                                        tempMV = (rx, ry, ref_frame_idx)
                    else:
                        if currFnum - ref_frame_idx > 0:
                            ref_f = self.scaled_frames[currFnum - ref_frame_idx - 1]
                        else:
                            ref_f = self.scaledBlackFrame
                        # print(ref_f)
                        for ry in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                            for rx in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                                ref_x = x * self.FMEEnable + rx
                                ref_y = y * self.FMEEnable + ry
                                # print("refxy: ", ref_x, ref_y)

                                if (
                                        0 <= ref_x + self.blockSize <= (
                                        self.widthV * self.FMEEnable - (self.FMEEnable - 1))
                                        and 0 <= ref_y + self.blockSize <= (
                                                self.heightV * self.FMEEnable - (self.FMEEnable - 1))
                                ) and (
                                        0 <= ref_x <= (self.widthV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                        and 0 <= ref_y <= (self.heightV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                ):
                                    refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                    # print(ref_f )
                                    maeT = self.MAE(currentBlock, refBlock)
                                    axy = np.abs(ry) + np.abs(rx)

                                    if maeT <= min_mae and axy <= min_axy:
                                        min_mae = maeT
                                        min_axy = axy
                                        tempMatch = refBlock
                                        tempMV = (rx, ry, ref_frame_idx)

                residual_block = currentBlock - tempMatch

                transed = dct_2d(residual_block)
                quantB = quantization(transed, QP)
                racB = racelling(quantB, QP)
                rtB = idct_2d(racB)
                recons_block = self.reconstructed_block(tempMatch, rtB)
                # recons_block = tempMatch + rtB

                resE = entropy_encoder(quantB)

                motion_V[i].append(tempMV)
                QTC_F[i].append(resE)
                reconstructed_frame[i].append(recons_block)

        self.reconstructedFrame[currFnum] = self.constractFrame(reconstructed_frame)
        motionV = motion_V
        # print(motionV)
        return [motionV, QTC_F]

    def Full_search_RC(self, currFnum, iRange, table, bit_frame, QP=6, nRefFrames=1):
        """
        _summary_

        Args:
            currFnum (int): Current frame number
            iRange (int): Search range for motion estimation
            QP (int): Quantization parameter
            nRefFrames (int): Number of reference frames
        """
        print("The", currFnum, "frame, P")
        blackframe = np.full((self.heightV, self.widthV), 128)

        # Initialize lists to store motion vectors and residual data for each block
        motion_V = []
        QTC_F = []
        reconstructed_frame = []

        # Get the current and reference frames
        curr_f = self.blockedYF[currFnum]
        ref_frames = []
        for i in range(nRefFrames):
            if currFnum - i > 0:
                ref_frames.append(self.reconstructedFrame[currFnum - i - 1])
            else:
                ref_frames.append(blackframe)
        # print(ref_frames)

        bitcount_row_array = []

        for i in range(self.blockNumInHeight):
            motion_V.append([])
            QTC_F.append([])
            reconstructed_frame.append([])
            bitcount_row = 0

            for j in range(self.blockNumInWidth):
                min_mae = float('inf')
                min_axy = float('inf')
                currentBlock = curr_f[i][j]
                tempMV = None
                tempMatch = None
                x = j * self.blockSize
                y = i * self.blockSize
                for ref_frame_idx, ref_f in enumerate(ref_frames):
                    # print(ref_f)
                    if self.FMEEnable == 1:
                        if currFnum - ref_frame_idx > 0:
                            ref_f = self.reconstructedFrame[currFnum - ref_frame_idx - 1]
                        else:
                            ref_f = blackframe
                        for ry in range(iRange, -iRange - 1, -1):
                            for rx in range(iRange, -iRange - 1, -1):
                                ref_x = x + rx
                                ref_y = y + ry
                                # print("refxy: ", ref_x, ref_y)

                                if (
                                        0 <= ref_x + self.blockSize <= self.widthV
                                        and 0 <= ref_y + self.blockSize <= self.heightV
                                ) and (
                                        0 <= ref_x <= self.widthV - 1
                                        and 0 <= ref_y <= self.heightV - 1
                                ):
                                    refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                    # print(ref_f )
                                    maeT = self.MAE(currentBlock, refBlock)
                                    axy = np.abs(ry) + np.abs(rx)

                                    if maeT <= min_mae and axy <= min_axy:
                                        min_mae = maeT
                                        min_axy = axy
                                        tempMatch = refBlock
                                        tempMV = (rx, ry, ref_frame_idx)
                    else:
                        if currFnum - ref_frame_idx > 0:
                            ref_f = self.scaled_frames[currFnum - ref_frame_idx - 1]
                        else:
                            ref_f = self.scaledBlackFrame
                        # print(ref_f)
                        for ry in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                            for rx in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                                ref_x = x * self.FMEEnable + rx
                                ref_y = y * self.FMEEnable + ry
                                # print("refxy: ", ref_x, ref_y)

                                if (
                                        0 <= ref_x + self.blockSize <= (
                                        self.widthV * self.FMEEnable - (self.FMEEnable - 1))
                                        and 0 <= ref_y + self.blockSize <= (
                                                self.heightV * self.FMEEnable - (self.FMEEnable - 1))
                                ) and (
                                        0 <= ref_x <= (self.widthV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                        and 0 <= ref_y <= (self.heightV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                ):
                                    refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                    # print(ref_f )
                                    maeT = self.MAE(currentBlock, refBlock)
                                    axy = np.abs(ry) + np.abs(rx)

                                    if maeT <= min_mae and axy <= min_axy:
                                        min_mae = maeT
                                        min_axy = axy
                                        tempMatch = refBlock
                                        tempMV = (rx, ry, ref_frame_idx)

                residual_block = currentBlock - tempMatch

                transed = dct_2d(residual_block)
                quantB = quantization(transed, QP)
                racB = racelling(quantB, QP)
                rtB = idct_2d(racB)
                recons_block = self.reconstructed_block(tempMatch, rtB)
                # recons_block = tempMatch + rtB

                resE = entropy_encoder(quantB)

                motion_V[i].append(tempMV)
                QTC_F[i].append(resE)
                reconstructed_frame[i].append(recons_block)
                #     # exercise 1
                #     bit_spend = tempMV[0].bit_length()+ tempMV[1].bit_length()+ tempMV[2].bit_length() +len(resE)*8
                #     # change QP
                #     bit_frame -= bit_spend
                # if self.blockNumInHeight-i-1 > 0:
                #     # print(bit_frame)
                #     bit_row = bit_frame // (self.blockNumInHeight - i - 1)
                #     min_difference = 100000000
                #     for key, value in table.items():
                #         difference = value - bit_row
                #         if 0 < difference < min_difference:
                #             min_difference = difference
                #             qp = key
                #     # print(qp)
                #     QP = qp
                # exercise 2
                bit_spend = tempMV[0].bit_length() + tempMV[1].bit_length() + tempMV[2].bit_length() + len(resE) * 8
                bitcount_row += bit_spend
            bitcount_row_array.append(bitcount_row)
        bitcount_row_array = np.array(bitcount_row_array)
        bitcount_frame = np.sum(bitcount_row_array)
        # calculate the scaling factor
        bitcount_frame_statistics = table[QP] * self.blockNumInHeight
        scaling_factor = bitcount_frame / bitcount_frame_statistics
        self.bitcount_scaling_factor[currFnum] = scaling_factor
        # calculate the bitcount proportion for every row
        bitcount_proportion_row = bitcount_row_array / bitcount_frame
        self.row_bitcount_proportion[currFnum] = bitcount_proportion_row

        self.reconstructedFrame[currFnum] = self.constractFrame(reconstructed_frame)
        motionV = motion_V
        # print(motionV)
        return [motionV, QTC_F]

    def Full_search_RC_2(self, currFnum, iRange, table, bit_frame, QP=6, nRefFrames=1):
        """
        _summary_
        this function is the search function in the second pass of the encoding process
        Args:
            currFnum (int): Current frame number
            iRange (int): Search range for motion estimation
            QP (int): Quantization parameter
            nRefFrames (int): Number of reference frames
        """
        print("The", currFnum, "frame, P")
        blackframe = np.full((self.heightV, self.widthV), 128)

        # Initialize lists to store motion vectors and residual data for each block
        motion_V = []
        QTC_F = []
        reconstructed_frame = []

        # Get the current and reference frames
        curr_f = self.blockedYF[currFnum]
        ref_frames = []
        for i in range(nRefFrames):
            if currFnum - i > 0:
                ref_frames.append(self.reconstructedFrame[currFnum - i - 1])
            else:
                ref_frames.append(blackframe)
        # print(ref_frames)

        # exercise 2 part c: refine the fixed table
        table = {0: 35839, 1: 28684, 2: 21232, 3: 14395, 4: 8542, 5: 4228, 6: 1589, 7: 687, 8: 466, 9: 416, 10: 410,
                 11: 409}
        scaling_factor = self.bitcount_scaling_factor[currFnum]
        for key in table:
            table[key] *= scaling_factor

        # exercise 2 part d: initialize a row-average-qp-value
        # qp_row_average = 0

        for i in range(self.blockNumInHeight):
            motion_V.append([])
            QTC_F.append([])
            reconstructed_frame.append([])

            # exercise 2
            # change QP
            bit_row = bit_frame * self.row_bitcount_proportion[currFnum][i]
            min_difference = 100000000
            for key, value in table.items():
                difference = value - bit_row
                if 0 < difference < min_difference:
                    min_difference = difference
                    qp = key
            QP = qp
            # qp_row_average += qp
            # print(QP)

            for j in range(self.blockNumInWidth):
                min_mae = float('inf')
                min_axy = float('inf')
                currentBlock = curr_f[i][j]
                tempMV = None
                tempMatch = None
                x = j * self.blockSize
                y = i * self.blockSize
                for ref_frame_idx, ref_f in enumerate(ref_frames):
                    # print(ref_f)
                    if self.FMEEnable == 1:
                        if currFnum - ref_frame_idx > 0:
                            ref_f = self.reconstructedFrame[currFnum - ref_frame_idx - 1]
                        else:
                            ref_f = blackframe
                        for ry in range(iRange, -iRange - 1, -1):
                            for rx in range(iRange, -iRange - 1, -1):
                                ref_x = x + rx
                                ref_y = y + ry
                                # print("refxy: ", ref_x, ref_y)

                                if (
                                        0 <= ref_x + self.blockSize <= self.widthV
                                        and 0 <= ref_y + self.blockSize <= self.heightV
                                ) and (
                                        0 <= ref_x <= self.widthV - 1
                                        and 0 <= ref_y <= self.heightV - 1
                                ):
                                    refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                    # print(ref_f )
                                    maeT = self.MAE(currentBlock, refBlock)
                                    axy = np.abs(ry) + np.abs(rx)

                                    if maeT <= min_mae and axy <= min_axy:
                                        min_mae = maeT
                                        min_axy = axy
                                        tempMatch = refBlock
                                        tempMV = (rx, ry, ref_frame_idx)
                    else:
                        if currFnum - ref_frame_idx > 0:
                            ref_f = self.scaled_frames[currFnum - ref_frame_idx - 1]
                        else:
                            ref_f = self.scaledBlackFrame
                        # print(ref_f)
                        for ry in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                            for rx in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                                ref_x = x * self.FMEEnable + rx
                                ref_y = y * self.FMEEnable + ry
                                # print("refxy: ", ref_x, ref_y)

                                if (
                                        0 <= ref_x + self.blockSize <= (
                                        self.widthV * self.FMEEnable - (self.FMEEnable - 1))
                                        and 0 <= ref_y + self.blockSize <= (
                                                self.heightV * self.FMEEnable - (self.FMEEnable - 1))
                                ) and (
                                        0 <= ref_x <= (self.widthV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                        and 0 <= ref_y <= (self.heightV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                ):
                                    refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                    # print(ref_f )
                                    maeT = self.MAE(currentBlock, refBlock)
                                    axy = np.abs(ry) + np.abs(rx)

                                    if maeT <= min_mae and axy <= min_axy:
                                        min_mae = maeT
                                        min_axy = axy
                                        tempMatch = refBlock
                                        tempMV = (rx, ry, ref_frame_idx)

                residual_block = currentBlock - tempMatch

                transed = dct_2d(residual_block)
                quantB = quantization(transed, QP)
                racB = racelling(quantB, QP)
                rtB = idct_2d(racB)
                recons_block = self.reconstructed_block(tempMatch, rtB)
                # recons_block = tempMatch + rtB

                resE = entropy_encoder(quantB)

                motion_V[i].append(tempMV)
                QTC_F[i].append(resE)
                reconstructed_frame[i].append(recons_block)

        # exercise 2 part d: output the qp_row_average to a file
        # qp_row_average = qp_row_average//self.blockNumInHeight
        # with open('frame_average_qp.txt', 'a') as file:
        #     file.write(str(qp_row_average) + '\n')

        self.reconstructedFrame[currFnum] = self.constractFrame(reconstructed_frame)
        motionV = motion_V
        # print(motionV)
        return [motionV, QTC_F]

    def Full_search_RC_3(self, currFnum, iRange, table, bit_frame, QP=6, nRefFrames=1):
        """
        _summary_
        this function is the search function in the second pass of the encoding process
        Args:
            currFnum (int): Current frame number
            iRange (int): Search range for motion estimation
            QP (int): Quantization parameter
            nRefFrames (int): Number of reference frames
        """
        print("The", currFnum, "frame, P")
        blackframe = np.full((self.heightV, self.widthV), 128)

        # Initialize lists to store motion vectors and residual data for each block
        motion_V = []
        QTC_F = []
        reconstructed_frame = []

        # Get the current and reference frames
        curr_f = self.blockedYF[currFnum]
        ref_frames = []
        for i in range(nRefFrames):
            if currFnum - i > 0:
                ref_frames.append(self.reconstructedFrame[currFnum - i - 1])
            else:
                ref_frames.append(blackframe)
        # print(ref_frames)

        # exercise 2 part c: refine the fixed table
        table = {0: 35839, 1: 28684, 2: 21232, 3: 14395, 4: 8542, 5: 4228, 6: 1589, 7: 687, 8: 466, 9: 416, 10: 410,
                 11: 409}
        scaling_factor = self.bitcount_scaling_factor[currFnum]
        for key in table:
            table[key] *= scaling_factor

        # exercise 2 part d: initialize a row-average-qp-value
        # qp_row_average = 0

        # exercise 2 part e: leveraging the information of the first pass
        # get the p_number of current frame
        p_number = -1
        for idx in range(currFnum):
            if self.iPer[idx] == 0:
                p_number += 1
        # get the MV information of the first pass
        MV_firstpass = self.MV_temp[p_number]

        for i in range(self.blockNumInHeight):
            motion_V.append([])
            QTC_F.append([])
            reconstructed_frame.append([])

            # exercise 2
            # change QP
            bit_row = bit_frame * self.row_bitcount_proportion[currFnum][i]
            min_difference = 100000000
            for key, value in table.items():
                difference = value - bit_row
                if 0 < difference < min_difference:
                    min_difference = difference
                    qp = key
            QP = qp
            # qp_row_average += qp
            # print(QP)

            for j in range(self.blockNumInWidth):
                min_mae = float('inf')
                min_axy = float('inf')
                currentBlock = curr_f[i][j]
                tempMV = None
                tempMatch = None
                x = j * self.blockSize
                y = i * self.blockSize
                # refine around the mv of the first pass
                mv_firstpass = MV_firstpass[i][j]
                x += mv_firstpass[0]
                y += mv_firstpass[1]
                for ref_frame_idx, ref_f in enumerate(ref_frames):
                    # print(ref_f)
                    if self.FMEEnable == 1:
                        if currFnum - ref_frame_idx > 0:
                            ref_f = self.reconstructedFrame[currFnum - ref_frame_idx - 1]
                        else:
                            ref_f = blackframe
                        # eliminate the search range to a smaller region around the mv of the first pass
                        for ry in range(2, -3, -1):
                            for rx in range(2, -3, -1):
                                ref_x = x + rx
                                ref_y = y + ry
                                # print("refxy: ", ref_x, ref_y)

                                if (
                                        0 <= ref_x + self.blockSize <= self.widthV
                                        and 0 <= ref_y + self.blockSize <= self.heightV
                                ) and (
                                        0 <= ref_x <= self.widthV - 1
                                        and 0 <= ref_y <= self.heightV - 1
                                ):
                                    refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                    # print(ref_f )
                                    maeT = self.MAE(currentBlock, refBlock)
                                    axy = np.abs(ry) + np.abs(rx)

                                    if maeT <= min_mae and axy <= min_axy:
                                        min_mae = maeT
                                        min_axy = axy
                                        tempMatch = refBlock
                                        tempMV = (rx, ry, ref_frame_idx)
                    else:
                        if currFnum - ref_frame_idx > 0:
                            ref_f = self.scaled_frames[currFnum - ref_frame_idx - 1]
                        else:
                            ref_f = self.scaledBlackFrame
                        # print(ref_f)
                        for ry in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                            for rx in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                                ref_x = x * self.FMEEnable + rx
                                ref_y = y * self.FMEEnable + ry
                                # print("refxy: ", ref_x, ref_y)

                                if (
                                        0 <= ref_x + self.blockSize <= (
                                        self.widthV * self.FMEEnable - (self.FMEEnable - 1))
                                        and 0 <= ref_y + self.blockSize <= (
                                                self.heightV * self.FMEEnable - (self.FMEEnable - 1))
                                ) and (
                                        0 <= ref_x <= (self.widthV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                        and 0 <= ref_y <= (self.heightV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                ):
                                    refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                    # print(ref_f )
                                    maeT = self.MAE(currentBlock, refBlock)
                                    axy = np.abs(ry) + np.abs(rx)

                                    if maeT <= min_mae and axy <= min_axy:
                                        min_mae = maeT
                                        min_axy = axy
                                        tempMatch = refBlock
                                        tempMV = (rx, ry, ref_frame_idx)

                residual_block = currentBlock - tempMatch

                transed = dct_2d(residual_block)
                quantB = quantization(transed, QP)
                racB = racelling(quantB, QP)
                rtB = idct_2d(racB)
                recons_block = self.reconstructed_block(tempMatch, rtB)
                # recons_block = tempMatch + rtB

                resE = entropy_encoder(quantB)

                motion_V[i].append(tempMV)
                QTC_F[i].append(resE)
                reconstructed_frame[i].append(recons_block)

        # exercise 2 part d: output the qp_row_average to a file
        # qp_row_average = qp_row_average//self.blockNumInHeight
        # with open('frame_average_qp.txt', 'a') as file:
        #     file.write(str(qp_row_average) + '\n')

        self.reconstructedFrame[currFnum] = self.constractFrame(reconstructed_frame)
        motionV = motion_V
        # print(motionV)
        return [motionV, QTC_F]

    def fast_search(self, currFnum, iRange, QP=6, nRefFrames=1):
        """
        _summary_

        Args:
            currFnum (int): Current frame number
            iRange (int): Search range for motion estimation
            QP (int): Quantization parameter
            nRefFrames (int): Number of reference frames
        """
        print("The", currFnum, "frame, P")
        blackframe = np.full((self.heightV, self.widthV), 128)

        # Initialize lists to store motion vectors and residual data for each block
        motion_V = []
        QTC_F = []
        reconstructed_frame = []

        # Get the current and reference frames
        curr_f = self.blockedYF[currFnum]
        ref_frames = []
        for i in range(nRefFrames):
            if currFnum - i > 0:
                ref_frames.append(self.reconstructedFrame[currFnum - i - 1])
            else:
                ref_frames.append(blackframe)
        # print(ref_frames)

        for i in range(self.blockNumInHeight):
            motion_V.append([])
            QTC_F.append([])
            reconstructed_frame.append([])

            for j in range(self.blockNumInWidth):
                min_mae = float('inf')
                min_axy = float('inf')
                currentBlock = curr_f[i][j]
                tempMV = None
                tempMatch = None

                for ref_frame_idx, ref_f in enumerate(ref_frames):
                    # print(ref_f)
                    org_x = j * self.blockSize
                    org_y = i * self.blockSize
                    ref_x = j * self.blockSize
                    ref_y = i * self.blockSize
                    rx = 0
                    ry = 0
                    rx_diff = 1
                    ry_diff = 1
                    # print("refxy: ", ref_x, ref_y)
                    while rx_diff != 0 and ry_diff != 0:
                        for offset in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
                            ref_x_pre = ref_x
                            ref_y_pre = ref_y
                            ref_x, ref_y = ref_x + offset[0], ref_y + offset[1]
                            if (
                                    0 <= ref_x + self.blockSize <= self.widthV
                                    and 0 <= ref_y + self.blockSize <= self.heightV
                            ) and (
                                    0 <= ref_x <= self.widthV
                                    and 0 <= ref_y <= self.heightV
                            ):
                                refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                                maeT = self.MAE(currentBlock, refBlock)
                                ry = ref_y - org_y
                                rx = ref_x - org_x
                                axy = np.abs(ry) + np.abs(rx)

                                if maeT <= min_mae and axy <= min_axy:
                                    min_mae = maeT
                                    min_axy = axy
                                    tempMatch = refBlock
                                    ref_y += ry
                                    ref_x += rx
                                    tempMV = (rx, ry, ref_frame_idx)
                        rx_diff = ref_x - ref_x_pre
                        ry_diff = ref_y - ref_y_pre

                residual_block = currentBlock - tempMatch

                transed = dct_2d(residual_block)
                quantB = quantization(transed, QP)
                racB = racelling(quantB, QP)
                rtB = idct_2d(racB)
                recons_block = self.reconstructed_block(tempMatch, rtB)
                # recons_block = tempMatch + rtB

                resE = entropy_encoder(quantB)

                motion_V[i].append(tempMV)
                QTC_F[i].append(resE)
                reconstructed_frame[i].append(recons_block)

        self.reconstructedFrame[currFnum] = self.constractFrame(reconstructed_frame)
        motionV = motion_V
        # print(motionV)
        return [motionV, QTC_F]

    '''
    new intra Varible block size
    '''

    def intra_Pred_V(self, currFnum, QP=6):
        '''
        EXE 4 b
        :param currFnum:
        :param QP:
        :return: [mode, approximated_res]
        '''
        print("The", currFnum, "frame, I")
        lam = 4 * 2 ** ((QP - 12) / 3)
        curr_f = self.blockedYF[currFnum]
        approximated_res = []  # initilize appriximated residule of the frame
        QTC_F = [[0 for _ in range(self.blockNumInWidth)] for _ in range(self.blockNumInHeight)]
        reconstructed_frame = []  # initilize reconstructed frame
        mode = []
        blackB = np.full((self.blockSize, self.blockSize), 128, np.uint8)
        VaribleBlockFlag = [[0 for _ in range(self.blockNumInWidth)] for _ in range(self.blockNumInHeight)]
        for i in range(self.blockNumInHeight):
            approximated_res.append([])
            reconstructed_frame.append([])
            mode.append([])
            for j in range(self.blockNumInWidth):

                # unVarible block
                currentBlock = curr_f[i][j]
                if j == 0:
                    predB0 = np.copy(blackB)
                else:
                    predB0 = np.copy(reconstructed_frame[i][j - 1])
                if i == 0:
                    predB1 = np.copy(blackB)
                else:
                    predB1 = np.copy(reconstructed_frame[i - 1][j])
                for x in range(self.blockSize):
                    for y in range(self.blockSize):
                        predB0[y][x] = predB0[y][-1]
                for x in range(self.blockSize):
                    for y in range(self.blockSize):
                        predB1[y][x] = predB1[-1][x]
                mae0 = mse(currentBlock, predB0)
                mae1 = mse(currentBlock, predB1)
                if mae0 > mae1:
                    predB_UV = predB1
                    modeUV = 1
                    # mode[i].append(1)
                else:
                    predB_UV = predB0
                    modeUV = 0

                # Varible block
                # residual_block = self.get_residual_block(currentBlock, predB)
                residual_block = currentBlock - predB_UV

                # Q4 new
                transed = dct_2d(residual_block)
                quantB = quantization(transed, QP)
                racB = racelling(quantB, QP)
                rtB = idct_2d(racB)
                recons_block = predB_UV + rtB
                resE = entropy_encoder(quantB)
                size_UV = len(resE)
                QTC_UV = resE
                # print("UV",recons_block)
                approximated_res[i].append(resE)

                '''
                varible size part
                '''
                ref_b = [[[[], []], [[], []]],
                         [[[], []], [[], []]]]
                # for i1 in range(2):
                #     for j1 in range(2):
                #         ref_b[i1].append([[], []])
                # Extract the last column
                right_column = predB0[:, -1]
                # Get the intermediate index of the column
                middle_index = len(right_column) // 2
                # print("Luck",middle_index)
                # Split the last column in half
                f_h = right_column[:middle_index]
                second_half = right_column[middle_index:]
                ref_b[0][0][0] = f_h
                ref_b[1][0][0] = second_half

                # extract the last row
                last_row = predB1[-1, :]
                middle_index_row = len(last_row) // 2
                # print("LUCK",middle_index_row)
                # split the last row
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
                        mae0 = self.MAE(Ori_VB[y2][x2], preSB0_V)
                        mae1 = self.MAE(Ori_VB[y2][x2], preSB1_V)
                        if mae0 > mae1:
                            predBV[y2][x2] = preSB1_V
                            Mode_VB[y2][x2] = 1
                        else:
                            predBV[y2][x2] = preSB0_V
                            Mode_VB[y2][x2] = 0

                        residual_Sb = Ori_VB[y2][x2] - predBV[y2][x2]

                        transedSB = dct_2d(residual_Sb)
                        quantSB = quantization(transedSB, QP - 1)
                        racSB = racelling(quantSB, QP - 1)
                        rtSB = idct_2d(racSB)
                        recons_sblock = predBV[y2][x2] + rtSB
                        resSubBlockEntropy = entropy_encoder(quantSB)
                        QTC_V[y2][x2] = resSubBlockEntropy
                        size_V += len(resSubBlockEntropy)

                        # print(resSubBlockEntropy)
                        Recon_VSB[y2][x2] = recons_sblock
                        if y2 + 1 < 2:
                            ref_b[y2 + 1][x2][1] = recons_sblock[-1, :]
                            # # extract the last row
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
                    VaribleBlockFlag[i][j] = 1
                    QTC_F[i][j] = QTC_V
                    reconstructed_frame[i].append(resR_sb)
                    mode[i].append(Mode_VB)
                else:
                    VaribleBlockFlag[i][j] = 0
                    QTC_F[i][j] = QTC_UV
                    reconstructed_frame[i].append(recons_block)
                    mode[i].append(modeUV)

                # todo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                # recons_block = self.reconstructed_block(predB, rtB)
                # reconstructed_frame[i].append(resR_sb)
                # reconstructed_frame[i].append(recons_block)
        self.VaribleBlockIndicators.append(VaribleBlockFlag)
        res = self.constractFrame(reconstructed_frame)
        # print(VaribleBlockFlag)
        self.reconstructedFrame[currFnum] = res
        cv2.imwrite('I FrameF.jpg', res)

        return [mode, QTC_F, approximated_res]

    def intra_Pred(self, currFnum, QP=6):
        '''
        EXE 4 b
        :param currFnum:
        :param QP:
        :return:
        '''
        print("The", currFnum, "frame, I")

        curr_f = self.blockedYF[currFnum]
        approximated_res = []  # initilize appriximated residule of the frame
        reconstructed_frame = []  # initilize reconstructed frame
        mode = []
        blackB = np.full((self.blockSize, self.blockSize), 128, np.uint8)

        for i in range(self.blockNumInHeight):
            approximated_res.append([])
            reconstructed_frame.append([])
            mode.append([])
            for j in range(self.blockNumInWidth):

                # unVarible block
                currentBlock = curr_f[i][j]
                if j == 0:
                    predB0 = np.copy(blackB)
                else:
                    predB0 = np.copy(reconstructed_frame[i][j - 1])
                if i == 0:
                    predB1 = np.copy(blackB)
                else:
                    predB1 = np.copy(reconstructed_frame[i - 1][j])
                for x in range(self.blockSize):
                    for y in range(self.blockSize):
                        predB0[y][x] = predB0[y][-1]
                for x in range(self.blockSize):
                    for y in range(self.blockSize):
                        predB1[y][x] = predB1[-1][x]
                mae0 = self.MAE(currentBlock, predB0)
                mae1 = self.MAE(currentBlock, predB1)
                if mae0 > mae1:
                    predB = predB1
                    mode[i].append(1)
                else:
                    predB = predB0
                    mode[i].append(0)

                # Varible block
                # residual_block = self.get_residual_block(currentBlock, predB)
                residual_block = currentBlock - predB

                # Q4 new
                transed = dct_2d(residual_block)
                quantB = quantization(transed, QP)

                racB = racelling(quantB, QP)
                rtB = idct_2d(racB)

                # recons_block = self.reconstructed_block(predB, rtB)
                recons_block = predB + rtB
                resE = entropy_encoder(quantB)
                approximated_res[i].append(resE)
                # print(quantB.shape)
                reconstructed_frame[i].append(recons_block)
        res = self.constractFrame(reconstructed_frame)
        self.reconstructedFrame[currFnum] = res
        # cv2.imwrite('I Frame.jpg', res)

        return [mode, approximated_res]

    def constractFrame(self, blocked):
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
        cv2.imshow('hello', res)
        cv2.waitKey(40)  # 'q' to quit
        return res

    def get_residual_block(self, current_block, predicted_block):

        return (current_block - predicted_block)  # % 256

    def to_a_file(self, file_name, block):
        # Save the array to a text file
        np.savetxt(file_name, block, delimiter=',', fmt='%d')

    def reconstructed_block(self, current_block, app_residual):
        return (current_block + app_residual)  # % 256

    def intra_V_decoder(self, frnum, MODE, qtcF):
        Flag = self.VaribleBlockIndicators[frnum]
        newF = [[] for _ in range(self.blockNumInHeight)]
        for _ in range(self.blockNumInHeight):
            newF[_] = [[] for _ in range(self.blockNumInWidth)]
        for y in range(self.blockNumInHeight):
            for x in range(self.blockNumInWidth):
                print("X:", x, "Y: ", y)
                if x == 0:
                    predB0 = np.full((self.blockSize, self.blockSize), 128, np.uint8)
                else:
                    predB0 = np.copy(newF[y][x - 1])
                    for xb in range(self.blockSize):
                        for yb in range(self.blockSize):
                            predB0[yb][xb] = predB0[yb][-1]

                if y == 0:
                    predB1 = np.full((self.blockSize, self.blockSize), 128, np.uint8)
                else:
                    predB1 = np.copy(newF[y - 1][x])
                    for xb in range(self.blockSize):
                        for yb in range(self.blockSize):
                            predB1[yb][xb] = predB1[-1][xb]

                if Flag[y][x] == 1:
                    modeB = MODE[y][x]
                    ref_b = [[[[], []], [[], []]],
                             [[[], []], [[], []]]]
                    # for i1 in range(2):
                    #     for j1 in range(2):
                    #         ref_b[i1].append([[], []])
                    # extract the last column
                    right_column = predB0[:, -1]
                    # get the intermidiem index
                    middle_index = self.blockSize // 2
                    # print("Luck",middle_index)
                    # split the last column
                    f_h = right_column[:middle_index]
                    second_half = right_column[middle_index:]
                    ref_b[0][0][0] = f_h
                    ref_b[1][0][0] = second_half

                    # extract last row
                    last_row = predB1[-1, :]
                    middle_index_row = len(last_row) // 2
                    # print("LUCK",middle_index_row)
                    # split the last row 
                    first_half_row = last_row[:middle_index_row]
                    second_half_row = last_row[middle_index_row:]
                    ref_b[0][0][1] = first_half_row
                    ref_b[0][1][1] = second_half_row
                    res_b = [[[[], []], [[], []]],
                             [[[], []], [[], []]]]
                    for y2 in range(2):
                        for x2 in range(2):
                            if modeB[y2][x2] == 0:
                                preSB = np.zeros((middle_index, middle_index))
                                print("X:", x, "Y: ", y, "XXXXXX", "x2:", x2, "y2", y2)
                                for x3 in range(middle_index):
                                    for y3 in range(middle_index):
                                        preSB[y3][x3] = ref_b[y2][x2][0][y3]
                            else:
                                preSB = np.zeros((middle_index, middle_index))
                                for x3 in range(middle_index):
                                    for y3 in range(middle_index):
                                        preSB[y3][x3] = ref_b[y2][x2][1][x3]

                            Temp = qtcF[y][x][y2][x2]
                            quantSB = decode_entropy(middle_index * middle_index, Temp)
                            racSB = racelling(quantSB, self.QP - 1)
                            rtSB = idct_2d(racSB)
                            resT = rtSB + preSB
                            res_b[y2][x2] = resT

                            if y2 + 1 < 2:
                                ref_b[y2 + 1][x2][1] = resT[-1, :]
                                # # extract the last row
                            if x2 + 1 < 2:
                                ref_b[y2][x2 + 1][0] = resT[:, -1]

                    result_upper = np.concatenate((res_b[0][0], res_b[0][1]), axis=1)
                    result_lower = np.concatenate((res_b[1][0], res_b[1][1]), axis=1)
                    newB = np.concatenate((result_upper, result_lower), axis=0)
                    newF[y][x] = newB

                if Flag[y][x] == 0:
                    modeB = MODE[y][x]
                    # print("y:", y, "x:", x, "mode: ", modeB, "frame:", IFindex, fr)
                    if modeB == 1:
                        predB = predB1
                    else:
                        predB = predB0
                    orgC = qtcF[y][x]
                    orgR = decode_entropy(self.blockSize * self.blockSize, orgC)
                    racB = racelling(orgR, self.QP)
                    rtB = idct_2d(racB)
                    newB = predB + rtB
                    newF[y][x] = newB
        newX = self.constractFrame(newF)
        return newX

    # def decoder(self, approximated_residual, mv_file, MODE, QP=6):
    def decoder(self, QTCCoeeff, MDiff, QP=6):
        iPer = MDiff[0]
        QTC_C = QTCCoeeff[0]
        MODE_en = MDiff[2]
        MODE = []
        for ele in MODE_en:
            res = Diff_Intra_Decoder(ele)
            MODE.append(res)

        MV_en = MDiff[1]
        # print(MV_en)
        mv_file = []
        for ele in MV_en:
            res = Diff_Deco_inter(ele)
            mv_file.append(res)

        PFindex = 0
        IFindex = 0
        blackframe = np.full((self.heightV, self.widthV), 128, dtype=np.uint8)
        decodedFrame = []
        blackB = np.full((self.blockSize, self.blockSize), 128, np.uint8)
        for fr in range(10):  # !!!!!!!!!!!!!!!!!!!!!
            print("decoding", fr, "frame")
            VBSF = self.VaribleBlockIndicators[fr]
            if iPer[fr] == 0:
                '''if fr == 0:
                    ref_f = blackframe
                else:
                    ref_f = decodedFrame[fr - 1]'''

                newF = [[] for _ in range(self.blockNumInHeight)]
                for _ in range(self.blockNumInHeight):
                    newF[_] = [[] for _ in range(self.blockNumInWidth)]

                for y in range(self.blockNumInHeight):
                    for x in range(self.blockNumInWidth):
                        rx = mv_file[PFindex][y][x][0]
                        ry = mv_file[PFindex][y][x][1]
                        ref_frame_index = mv_file[PFindex][y][x][2]
                        if fr - ref_frame_index > 0:
                            ref_f = decodedFrame[fr - 1 - ref_frame_index]
                        else:
                            ref_f = blackframe
                        if self.FMEEnable == 1:
                            ref_x = int(x * self.blockSize + rx)
                            ref_y = int(y * self.blockSize + ry)
                            refB = ref_f[ref_y:ref_y + self.blockSize,
                                   ref_x:ref_x + self.blockSize]
                        else:
                            refB = self.generate_interpolated_reference_block(x, y, rx, ry, ref_f)

                        orgC = QTC_C[fr][y][x]

                        orgR = decode_entropy(self.blockSize * self.blockSize, orgC)
                        racB = racelling(orgR, QP)
                        rtB = idct_2d(racB)
                        newB = refB + rtB
                        newF[y][x] = newB

                newX = self.constractFrame(newF)
                decodedFrame.append(newX)
                PFindex += 1
            else:
                if self.VBSEnable == 1:
                    self.intra_V_decoder(fr, self.MODE[fr], MODE[IFindex])
                    IFindex += 1

                else:
                    newF = [[] for _ in range(self.blockNumInHeight)]
                    for _ in range(self.blockNumInHeight):
                        newF[_] = [[] for _ in range(self.blockNumInWidth)]
                    for y in range(self.blockNumInHeight):
                        for x in range(self.blockNumInWidth):
                            modeB = MODE[IFindex][y][x]
                            # print("y:", y, "x:", x, "mode: ", modeB, "frame:", IFindex, fr)
                            if x == 0:
                                predB0 = np.full((self.blockSize, self.blockSize), 128, np.uint8)
                            else:
                                predB0 = np.copy(newF[y][x - 1])
                                for xb in range(self.blockSize):
                                    for yb in range(self.blockSize):
                                        predB0[yb][xb] = predB0[yb][-1]

                            if y == 0:
                                predB1 = np.full((self.blockSize, self.blockSize), 128, np.uint8)
                            else:
                                predB1 = np.copy(newF[y - 1][x])
                                for xb in range(self.blockSize):
                                    for yb in range(self.blockSize):
                                        predB1[yb][xb] = predB1[-1][xb]

                            if modeB == 1:
                                predB = predB1
                            else:
                                predB = predB0

                            orgC = QTC_C[fr][y][x]
                            orgR = decode_entropy(self.blockSize * self.blockSize, orgC)
                            racB = racelling(orgR, QP)
                            rtB = idct_2d(racB)
                            newB = predB + rtB
                            newF[y][x] = newB
                    newX = self.constractFrame(newF)
                    decodedFrame.append(newX)
                    IFindex += 1

        # cv2.imshow('decoded', newX)
        # cv2.waitKey(880)  # 'q' to quit
        return decodedFrame

    def compare(decoded_frame, reconstructed_frame):
        # Compute the absolute difference between the two frames
        diff = cv2.absdiff(decoded_frame, reconstructed_frame)
        # Check if the frames fully match by checking if the difference is zero
        return np.sum(diff) == 0

    def encoder(self, blockSize=4, searchRange=16, QP=6, FMEEnable=1, VBSEnable=1, FastME=1, nReferenceframe=1):
        '''

        :param blockSize:
        :param searchRange:
        :param QP:
        :param FMEEnable:
        :param VBSEnable:
        :param FastME:
        :param nReferenceframe:
        :return:  QTCCoeeff = [self.QTCC]
                    MDiff = [self.iPer, diff_inter_lis, diff_intra_lis, self.VaribleBlockIndicators]  # self.VaribleBlockIndicators
        '''
        self.QP = QP
        self.scaledBlackFrame = np.full((575, 703), 128, np.uint8)
        self.VBSEnable = VBSEnable
        self.reconstructedFrame = [[] for _ in range(self.frameNum)]
        self.MV = []
        self.QTCC = []
        self.MODE = []
        self.VaribleBlockIndicators = []
        self.blockSpliting(blockSize)
        # initialize the FMEEnable parameter and the scaled frames
        self.FMEEnable = FMEEnable
        self.scaled_frames = []
        for i in range(21):  # !!!!!!!!!!!!!!!!!!!!!
            # for i in range(1):
            if self.iPer[i] == 0:
                # todo change this to New_search
                # res = self.Full_search(i, searchRange, QP, nRefFrames=4)
                if VBSEnable == 1:
                    res = self.New_search(i, searchRange, QP, nReferenceframe, FastME)
                    self.MV.append(res[0])
                    self.QTCC.append(res[1])
                else:
                    if FastME == 1:
                        res = self.fast_search(i, searchRange, QP, nReferenceframe)
                    else:
                        res = self.Full_search(i, searchRange, QP, nReferenceframe)
                    self.MV.append(res[0])
                    self.QTCC.append(res[1])

            else:
                if VBSEnable == 1:
                    res = self.intra_Pred_V(i)
                    self.MODE.append(res[0])
                    self.QTCC.append(res[1])
                else:
                    res = self.intra_Pred(i)
                    self.MODE.append(res[0])
                    self.QTCC.append(res[1])
            # save the scaled frames
            if self.FMEEnable != 1:
                self.scaled_frames.append(self.generate_scaled_ref_frames(self.reconstructedFrame[i]))
        # Decode MVs
        if VBSEnable == 0:
            diff_inter_lis = []
            for i in self.MV:
                # print(i)
                diff_inter = Diff_Enco_inter(i)
                # print(diff_inter)
                diff_inter_lis.append(diff_inter)
            diff_intra_lis = []
            for i in self.MODE:
                diff_intra = Diff_Intra_Encoder(i)
                diff_intra_lis.append(diff_intra)
        else:
            indI = 0
            indP = 0
            diff_inter_lis = []
            diff_intra_lis = []
            for i in range(len(self.VaribleBlockIndicators)):
                if self.iPer[i] == 0:
                    res = self.diff_encode_inter_perVBS(self.MV[indP], self.VaribleBlockIndicators[i])
                    diff_inter_lis.append(res)
                    indP += 1
                if self.iPer[i] == 1:
                    res = self.diff_encode_intra_perVBS(self.MODE[indI], self.VaribleBlockIndicators[i])
                    diff_intra_lis.append(res)
                    indI += 1

        QTCCoeeff = [self.QTCC]
        MDiff = [self.iPer, diff_inter_lis, diff_intra_lis, self.VaribleBlockIndicators]  # self.VaribleBlockIndicators
        MDiff = [self.iPer, self.MV, self.MODE, self.VaribleBlockIndicators]
        '''
        # packed_data1 = pkl.dumps(QTCCoeeff)
        # packed_data2 = pkl.dumps(MDiff)
        # with open('data1.pickle', 'wb') as file1:
        #     pkl.dump(packed_data1, file1)
        # with open('data.pickle', 'wb') as file2:
        #     pkl.dump(packed_data2, file2)
        '''
        return [QTCCoeeff, MDiff]

    def bit_count_row(self, block_size, QP):
        res = self.encoder(16, block_size, QP, 1, 0, 0, 1)
        iNum = 0
        pNum = 0
        row_bit = []
        for i in range(10):
            if i % 21 != 0:
                pNum += 1
                QTCC = res[0][0][i]
                MDiff1 = res[1][0][i]
                MDiff2 = res[1][1][pNum - 1]
                MDiff3 = []
            else:
                QTCC = res[0][0][i]
                MDiff1 = res[1][0][i]
                MDiff2 = []
                MDiff3 = res[1][2][iNum - 1]
            MDiff = [MDiff1, MDiff2, MDiff3]
            output = [QTCC, MDiff]
            EncoderOut2(output)

            if os.path.exists('QTCC.npy') and os.path.exists('MDiff.npy'):
                frame_bytescount = os.path.getsize('QTCC.npy')  # + os.path.getsize('MDiff.npy')
            frame_bitscount = frame_bytescount * 8
            row_bitcount = frame_bitscount / math.ceil(self.heightV / block_size)
            # row_bitcount = frame_bitscount/18
            row_bit.append(row_bitcount)
        filtered_list = [value for index, value in enumerate(row_bit) if index % 21 != 0]
        return round(sum(filtered_list) / len(filtered_list))

        '''if id_frame % 8 != 0:
            print('p frame')
            QTCC = res[0][0][id_frame]
            MDiff1 = res[1][0][id_frame]
            MDiff2 = res[1][1][pNum - 1]
            MDiff3 = []
            
        else:
            print('i frame')
            QTCC = res[0][0][id_frame]
            MDiff1 = res[1][0][id_frame]
            MDiff2 = []
            MDiff3 = res[1][2][iNum - 1]
        MDiff = [MDiff1, MDiff2, MDiff3]
        output = [QTCC, MDiff]
        EncoderOut2(output)
    
        if os.path.exists('QTCC.npy') and os.path.exists('MDiff.npy'):
            frame_bytescount = os.path.getsize('QTCC.npy') + os.path.getsize('MDiff.npy')
        frame_bitscount = frame_bytescount * 8
        row_bitcount = frame_bitscount/self.blockNumInHeight
        row_bit.append(row_bitcount)
    filtered_list = [value for index, value in enumerate(row_bit) if index % 8 != 0]
    print(round(sum(filtered_list) / len(filtered_list)))'''

    def bit_table(self, block_size, QP):
        table = {}
        for i in QP:
            bit = self.bit_count_row(block_size, i)
            table[i] = bit
        return table

    def RC_encoder(self, RCflag, targetBR, fps=30, blockSize=4, searchRange=4, QP=6, QPs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   FMEEnable=1, VBSEnable=1, FastME=1, nReferenceframe=1):
        if RCflag == 0:
            return self.encoder(blockSize, searchRange, QP, FMEEnable, VBSEnable, FastME, nReferenceframe)
        else:
            self.blockSpliting(blockSize)
            bit_frame = targetBR // fps
            bit_row = bit_frame // self.blockNumInHeight
            # tabel = self.bit_table(blockSize,QPs)
            tabel = {0: 35839, 1: 28684, 2: 21232, 3: 14395, 4: 8542, 5: 4228, 6: 1589, 7: 687, 8: 466, 9: 416, 10: 410,
                     11: 409}
            # tabel = {0: 17056, 1: 13460, 2: 10208, 3: 7505, 4: 5051, 5: 2962, 6: 1118, 7: 480, 8: 322, 9: 291, 10: 290, 11: 290}
            print("bit table is", tabel)
            qp = None
            min_difference = float('inf')

            for key, value in tabel.items():
                difference = value - bit_row
                if 0 < difference < min_difference:
                    min_difference = difference
                    qp = key
            print("best qp is:", qp)

            # return self.encoder(blockSize, searchRange, qp, FMEEnable, VBSEnable, FastME, nReferenceframe)
            self.QP = qp
            self.scaledBlackFrame = np.full((575, 703), 128, np.uint8)
            self.VBSEnable = VBSEnable
            self.reconstructedFrame = [[] for _ in range(self.frameNum)]
            self.MV = []
            self.QTCC = []
            self.MODE = []
            self.VaribleBlockIndicators = []
            # initialize the FMEEnable parameter and the scaled frames
            self.FMEEnable = FMEEnable
            self.scaled_frames = []
            # initialize an array to store the proportion of bitcount spent per row
            self.row_bitcount_proportion = [[] for _ in range(self.frameNum)]
            # initialize an array to store the bitcount scaling factor
            self.bitcount_scaling_factor = [1 for _ in range(self.frameNum)]

            # exercise 2 part d: read qp value from the file
            file_path = 'frame_average_qp.txt'
            with open(file_path, 'r') as file:
                lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            the_refined_qp_values = []
            for i in range(len(numbers)):
                if i == 0:
                    the_refined_qp_values.append(qp)
                else:
                    the_refined_qp_values.append(numbers[i - 1])

            for i in range(21):  # !!!!!!!!!!!!!!!!!!!!!
                if RCflag != 1:
                    qp_per_frame = the_refined_qp_values[i]
                else:
                    qp_per_frame = qp
                if self.iPer[i] == 0:
                    # todo change this to New_search
                    # res = self.Full_search(i, searchRange, QP, nRefFrames=4)
                    if VBSEnable == 1:
                        res = self.New_search(i, searchRange, qp_per_frame, nReferenceframe, FastME)
                        self.MV.append(res[0])
                        self.QTCC.append(res[1])
                    else:
                        if FastME == 1:
                            res = self.fast_search(i, searchRange, qp_per_frame, nReferenceframe)
                        else:
                            res = self.Full_search_RC(i, searchRange, tabel, bit_frame, qp_per_frame, nReferenceframe)
                            # res = self.Full_search(i, searchRange, QP, nReferenceframe)
                        self.MV.append(res[0])
                        self.QTCC.append(res[1])

                else:
                    if VBSEnable == 1:
                        res = self.intra_Pred_V(i, qp_per_frame)
                        self.MODE.append(res[0])
                        self.QTCC.append(res[1])
                    else:
                        res = self.intra_Pred(i, qp_per_frame)
                        self.MODE.append(res[0])
                        self.QTCC.append(res[1])
                # save the scaled frames
                if self.FMEEnable != 1:
                    self.scaled_frames.append(self.generate_scaled_ref_frames(self.reconstructedFrame[i]))
            # Decode MVs
            if VBSEnable == 0:
                diff_inter_lis = []
                for i in self.MV:
                    # print(i)
                    diff_inter = Diff_Enco_inter(i)
                    # print(diff_inter)
                    diff_inter_lis.append(diff_inter)
                diff_intra_lis = []
                for i in self.MODE:
                    diff_intra = Diff_Intra_Encoder(i)
                    diff_intra_lis.append(diff_intra)
            else:
                indI = 0
                indP = 0
                diff_inter_lis = []
                diff_intra_lis = []
                for i in range(len(self.VaribleBlockIndicators)):
                    if self.iPer[i] == 0:
                        res = self.diff_encode_inter_perVBS(self.MV[indP], self.VaribleBlockIndicators[i])
                        diff_inter_lis.append(res)
                        indP += 1
                    if self.iPer[i] == 1:
                        res = self.diff_encode_intra_perVBS(self.MODE[indI], self.VaribleBlockIndicators[i])
                        diff_intra_lis.append(res)
                        indI += 1

            QTCCoeeff = [self.QTCC]
            MDiff = [self.iPer, diff_inter_lis, diff_intra_lis,
                     self.VaribleBlockIndicators]  # self.VaribleBlockIndicators
            MDiff = [self.iPer, self.MV, self.MODE, self.VaribleBlockIndicators]

            res = [QTCCoeeff, MDiff]
            if RCflag == 1:
                return res
            else:
                # second pass encoding
                # a:change the prediction mode of scene changes to I frame
                print("start second pass encoding")
                self.frame_bitscount = []
                iNum = 0
                pNum = 0
                for i in range(21):
                    if self.iPer[i] != 0:
                        iNum += 1
                        QTCC = res[0][0][i]
                        MDiff1 = res[1][0][i]
                        MDiff2 = []
                        MDiff3 = res[1][2][iNum - 1]
                    else:
                        pNum += 1
                        QTCC = res[0][0][i]
                        MDiff1 = res[1][0][i]
                        MDiff2 = res[1][1][pNum - 1]
                        MDiff3 = []
                    MDiff = [MDiff1, MDiff2, MDiff3]
                    output = [QTCC, MDiff]
                    EncoderOut2(output)

                    if os.path.exists('QTCC.npy') and os.path.exists('MDiff.npy'):
                        frame_bytescount = os.path.getsize('QTCC.npy') + os.path.getsize('MDiff.npy')
                    frame_bitscount = frame_bytescount * 8
                    self.frame_bitscount.append(frame_bitscount)
                scene_change_threshold = {0: 869000, 1: 702000, 2: 554000, 3: 424000, 4: 311000, 5: 217000,
                                          6: 141000, 7: 83000, 8: 43000, 9: 21000, 10: 18000, 11: 32000}
                for i in range(21):
                    if self.iPer[i] == 0:
                        pFrame_bitscount = self.frame_bitscount[i]
                        if pFrame_bitscount > scene_change_threshold[qp]:
                            # encode as an I frame
                            self.iPer[i] = 1
                # second pass encoding, using different functions
                # clear the QTCC and MV and MODE
                self.MV_temp = self.MV
                self.MV = []
                self.MODE = []
                self.QTCC = []
                for i in range(21):  # !!!!!!!!!!!!!!!!!!!!!
                    if self.iPer[i] == 0:
                        # todo change this to New_search
                        # res = self.Full_search(i, searchRange, QP, nRefFrames=4)
                        if VBSEnable == 1:
                            res = self.New_search(i, searchRange, qp, nReferenceframe, FastME)
                            self.MV.append(res[0])
                            self.QTCC.append(res[1])
                        else:
                            if FastME == 1:
                                res = self.fast_search(i, searchRange, qp, nReferenceframe)
                            else:
                                if RCflag == 2:
                                    # using different function
                                    res = self.Full_search_RC_2(i, searchRange, tabel, bit_frame, qp, nReferenceframe)
                                    # res = self.Full_search(i, searchRange, QP, nReferenceframe)
                                else:
                                    res = self.Full_search_RC_3(i, searchRange, tabel, bit_frame, qp, nReferenceframe)
                            self.MV.append(res[0])
                            self.QTCC.append(res[1])

                    else:
                        if VBSEnable == 1:
                            res = self.intra_Pred_V(i, qp)
                            self.MODE.append(res[0])
                            self.QTCC.append(res[1])
                        else:
                            res = self.intra_Pred(i, qp)
                            self.MODE.append(res[0])
                            self.QTCC.append(res[1])
                    # save the scaled frames
                    if self.FMEEnable != 1:
                        self.scaled_frames.append(self.generate_scaled_ref_frames(self.reconstructedFrame[i]))
                # Decode MVs
                if VBSEnable == 0:
                    diff_inter_lis = []
                    for i in self.MV:
                        # print(i)
                        diff_inter = Diff_Enco_inter(i)
                        # print(diff_inter)
                        diff_inter_lis.append(diff_inter)
                    diff_intra_lis = []
                    for i in self.MODE:
                        diff_intra = Diff_Intra_Encoder(i)
                        diff_intra_lis.append(diff_intra)
                else:
                    indI = 0
                    indP = 0
                    diff_inter_lis = []
                    diff_intra_lis = []
                    for i in range(len(self.VaribleBlockIndicators)):
                        if self.iPer[i] == 0:
                            res = self.diff_encode_inter_perVBS(self.MV[indP], self.VaribleBlockIndicators[i])
                            diff_inter_lis.append(res)
                            indP += 1
                        if self.iPer[i] == 1:
                            res = self.diff_encode_intra_perVBS(self.MODE[indI], self.VaribleBlockIndicators[i])
                            diff_intra_lis.append(res)
                            indI += 1

                QTCCoeeff = [self.QTCC]
                MDiff = [self.iPer, diff_inter_lis, diff_intra_lis,
                         self.VaribleBlockIndicators]  # self.VaribleBlockIndicators
                MDiff = [self.iPer, self.MV, self.MODE, self.VaribleBlockIndicators]

                res = [QTCCoeeff, MDiff]
                return res

    def newBlocksearchP(self, ix, iy, block_size, iRange, nRefFrames, currFnum, QP):
        '''
        res[0]=> resconstruct block
        [1] mv
        [2] residual block
        [3] res size:
        :param: ix and iy are the index of the current block
        :param block_size: the current block size
        :param iRange: the search range
        :param nRefFrames: how many reference frames we are going to use
        :param currFnum: the index of current frame
        :param QP: the quantization parameter
        :return:
        '''
        # gain the current block
        current_frame = self.yFrame[currFnum]
        x = ix
        y = iy
        block = current_frame[y:y + block_size, x:x + block_size]
        min_mae = 114514
        min_axy = 114514
        tempMV = None
        tempMatch = None
        blackframe = np.full((self.heightV, self.widthV), 128)
        # iterate all reference frames
        for i in range(nRefFrames):
            if self.FMEEnable == 1:
                if currFnum - i > 0:
                    ref_f = self.reconstructedFrame[currFnum - i - 1]
                else:
                    ref_f = blackframe
                for ry in range(iRange, -iRange - 1, -1):
                    for rx in range(iRange, -iRange - 1, -1):
                        ref_x = x + rx
                        ref_y = y + ry
                        # print("refxy: ", ref_x, ref_y)

                        if (
                                0 <= ref_x + block_size <= self.widthV
                                and 0 <= ref_y + block_size <= self.heightV
                        ) and (
                                0 <= ref_x <= self.widthV - 1
                                and 0 <= ref_y <= self.heightV - 1
                        ):
                            refBlock = ref_f[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                            # print(ref_f )
                            maeT = self.MAE(block, refBlock)
                            axy = np.abs(ry) + np.abs(rx)

                            if maeT <= min_mae and axy <= min_axy:
                                min_mae = maeT
                                min_axy = axy
                                tempMatch = refBlock
                                tempMV = (rx, ry, i)
            else:
                if currFnum - i > 0:
                    ref_f = self.scaled_frames[currFnum - i - 1]
                else:
                    ref_f = self.scaledBlackFrame
                # print(ref_f)
                for ry in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                    for rx in range(iRange * self.FMEEnable, -iRange * self.FMEEnable - 1, -1):
                        ref_x = x * self.FMEEnable + rx
                        ref_y = y * self.FMEEnable + ry
                        # print("refxy: ", ref_x, ref_y)

                        if (
                                0 <= ref_x + block_size <= (self.widthV * self.FMEEnable - (self.FMEEnable - 1))
                                and 0 <= ref_y + block_size <= (self.heightV * self.FMEEnable - (self.FMEEnable - 1))
                        ) and (
                                0 <= ref_x <= (self.widthV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                and 0 <= ref_y <= (self.heightV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                        ):
                            refBlock = ref_f[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                            # print(ref_f )
                            maeT = self.MAE(block, refBlock)
                            axy = np.abs(ry) + np.abs(rx)

                            if maeT <= min_mae and axy <= min_axy:
                                min_mae = maeT
                                min_axy = axy
                                tempMatch = refBlock
                                tempMV = (rx, ry, i)
        residual_block = block - tempMatch

        transed = dct_2d(residual_block)
        quantB = quantization(transed, QP)
        racB = racelling(quantB, QP)
        rtB = idct_2d(racB)
        recons_block = self.reconstructed_block(tempMatch, rtB)

        resE = entropy_encoder(quantB)

        # calculate the size of resE and MV in bytes
        size = sys.getsizeof(resE) + sys.getsizeof(tempMV)
        return [recons_block, tempMV, resE, size]

    def fast_newBlocksearchP(self, ix, iy, block_size, iRange, nRefFrames, currFnum, QP):
        '''
        res[0]=> resconstruct block
        [1] mv
        [2] residual block
        [3] res size:
        :param: ix and iy are the index of the current block
        :param block_size: the current block size
        :param iRange: the search range
        :param nRefFrames: how many reference frames we are going to use
        :param currFnum: the index of current frame
        :param QP: the quantization parameter
        :return:
        '''
        # gain the current block
        current_frame = self.yFrame[currFnum]
        x = ix
        y = iy
        block = current_frame[y:y + block_size, x:x + block_size]
        min_mae = 114514
        min_axy = 114514
        tempMV = None
        tempMatch = None
        blackframe = np.full((self.heightV, self.widthV), 128)
        # iterate all reference frames
        for i in range(nRefFrames):
            if self.FMEEnable == 1:
                if currFnum - i > 0:
                    ref_f = self.reconstructedFrame[currFnum - i - 1]
                else:
                    ref_f = blackframe

                org_x = x
                org_y = y
                ref_x = x
                ref_y = y
                rx = 0
                ry = 0
                rx_diff = 1
                ry_diff = 1
                # print("refxy: ", ref_x, ref_y)
                while rx_diff != 0 and ry_diff != 0:
                    for offset in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
                        ref_x_pre = ref_x
                        ref_y_pre = ref_y
                        ref_x, ref_y = ref_x + offset[0], ref_y + offset[1]
                        if (
                                0 <= ref_x + self.blockSize <= self.widthV
                                and 0 <= ref_y + self.blockSize <= self.heightV
                        ) and (
                                0 <= ref_x <= self.widthV
                                and 0 <= ref_y <= self.heightV
                        ):
                            refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                            maeT = self.MAE(block, refBlock)
                            ry = ref_y - org_y
                            rx = ref_x - org_x
                            axy = np.abs(ry) + np.abs(rx)

                            if maeT <= min_mae and axy <= min_axy:
                                min_mae = maeT
                                min_axy = axy
                                tempMatch = refBlock
                                ref_y += ry
                                ref_x += rx
                                tempMV = (rx, ry, i)
                    rx_diff = ref_x - ref_x_pre
                    ry_diff = ref_y - ref_y_pre
            else:
                if currFnum - i > 0:
                    ref_f = self.scaled_frames[currFnum - i - 1]
                else:
                    ref_f = self.scaledBlackFrame
                org_x = x * self.FMEEnable
                org_y = y * self.FMEEnable
                ref_x = x * self.FMEEnable
                ref_y = y * self.FMEEnable
                rx = 0
                ry = 0
                rx_diff = 1
                ry_diff = 1
                # print("refxy: ", ref_x, ref_y)
                while rx_diff != 0 and ry_diff != 0:
                    for offset in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
                        ref_x_pre = ref_x
                        ref_y_pre = ref_y
                        ref_x, ref_y = ref_x + offset[0], ref_y + offset[1]
                        if (
                                0 <= ref_x + block_size <= (self.widthV * self.FMEEnable - (self.FMEEnable - 1))
                                and 0 <= ref_y + block_size <= (self.heightV * self.FMEEnable - (self.FMEEnable - 1))
                        ) and (
                                0 <= ref_x <= (self.widthV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                                and 0 <= ref_y <= (self.heightV * self.FMEEnable - (self.FMEEnable - 1)) - 1
                        ):
                            refBlock = ref_f[ref_y:ref_y + self.blockSize, ref_x:ref_x + self.blockSize]
                            maeT = self.MAE(block, refBlock)
                            ry = ref_y - org_y
                            rx = ref_x - org_x
                            axy = np.abs(ry) + np.abs(rx)

                            if maeT <= min_mae and axy <= min_axy:
                                min_mae = maeT
                                min_axy = axy
                                tempMatch = refBlock
                                ref_y += ry
                                ref_x += rx
                                tempMV = (rx, ry, i)
                    rx_diff = ref_x - ref_x_pre
                    ry_diff = ref_y - ref_y_pre

        residual_block = block - tempMatch

        transed = dct_2d(residual_block)
        quantB = quantization(transed, QP)
        racB = racelling(quantB, QP)
        rtB = idct_2d(racB)
        recons_block = self.reconstructed_block(tempMatch, rtB)

        resE = entropy_encoder(quantB)

        # calculate the size of resE and MV in bytes
        size = sys.getsizeof(resE) + sys.getsizeof(tempMV)
        return [recons_block, tempMV, resE, size]

    def varibleBS_block(self, x, y, frame_num, iRange, QP, nRefFrames, FastME):  # todo irange
        # print("VBS:",x,y)
        const = 2
        lam = 0.2 * 2 ** ((QP - 12) / 3)
        # print("fr:", frame_num)
        curr_f = self.blockedYF[frame_num]

        # Unsplit block
        curr_b = curr_f[y][x]

        #  def newBlocksearchP(self, x, y, block_size, iRange, nRefFrames, currFnum, QP):
        #
        if FastME == 1:
            res = self.fast_newBlocksearchP(x * self.blockSize, y * self.blockSize, self.blockSize, iRange, nRefFrames,
                                            frame_num, QP)
        else:
            res = self.newBlocksearchP(x * self.blockSize, y * self.blockSize, self.blockSize, iRange, nRefFrames,
                                       frame_num, QP)
        # print("XXXXXXXXXX",res)
        UnSplitRes = res[0]
        UnSplitMV = res[1]
        UnsplitEnres = res[2]
        UnSplitsize = res[3] + 2

        # Spilit block
        spilitRes = [[0, 0], [0, 0]]
        splitMV = [[0, 0], [0, 0]]
        splitEnres = [[0, 0], [0, 0]]
        spilitsize = 8
        sub_block_size = self.blockSize // 2
        for i in range(2):
            for j in range(2):
                # sub_block_c = [row[j * sub_block_size:(j + 1) * sub_block_size] for row in
                #                curr_b[i * sub_block_size:(i + 1) * sub_block_size]]
                sbindex_x = x * self.blockSize + j * sub_block_size
                sbindex_y = y * self.blockSize + i * sub_block_size
                # print("VBS:",VBs[i][j])
                #  def newBlocksearchP(self, x, y, block_size, iRange, nRefFrames, currFnum, QP):
                res = self.newBlocksearchP(sbindex_x, sbindex_y, sub_block_size, iRange, nRefFrames, frame_num, QP)
                # print("RES:",res)
                spilitRes[i][j] = res[0]
                splitEnres[i][j] = res[2]
                splitMV[i][j] = res[1]
                spilitsize += res[3]

        result_upper = np.concatenate((spilitRes[0][0], spilitRes[0][1]), axis=1)
        result_lower = np.concatenate((spilitRes[1][0], spilitRes[1][1]), axis=1)
        spilitResConstructed = np.concatenate((result_upper, result_lower), axis=0)

        D_us = mse(UnSplitRes, curr_b)
        # print('D_us' , D_us)
        J_us = D_us + lam * UnSplitsize

        D_s = mse(spilitResConstructed, curr_b)
        # print('D_s', D_s)
        J_s = D_s + lam * spilitsize
        # print(J_s, J_us)
        if J_s < J_us:
            self.VaribleBlockIndicators[frame_num][y][x] = 1
            recons_block = spilitResConstructed
            mv = splitMV
            resE = splitEnres
        else:
            self.VaribleBlockIndicators[frame_num][y][x] = 0
            recons_block = UnSplitRes
            mv = UnSplitMV
            resE = UnsplitEnres

        return [recons_block, mv, resE]

    def New_search(self, currFnum, iRange=4, QP=6, nRefFrames=1, FastME=1):
        """
        _summary_

        Args:
            currFnum (int): Current frame number
            iRange (int): Search range for motion estimation
            QP (int): Quantization parameter
            nRefFrames (int): Number of reference frames
        """
        print("The", currFnum, "frame, P")
        VaribleBlockIndicator = [[0 for _ in range(self.blockNumInWidth)] for _ in range(self.blockNumInHeight)]
        # Initialize lists to store motion vectors and residual data for each block
        self.VaribleBlockIndicators.append(VaribleBlockIndicator)
        # print("LENNN::", len(self.VaribleBlockIndicators))
        motion_V = []
        QTC_F = []
        reconstructed_frame = []

        # Get the current frame
        curr_f = self.blockedYF[currFnum]

        for i in range(self.blockNumInHeight):
            motion_V.append([])
            QTC_F.append([])
            reconstructed_frame.append([])

            for j in range(self.blockNumInWidth):
                # def varibleBS_block(self, x, y, frame_num, iRange, QP, nRefFrames):  # todo irange
                res = self.varibleBS_block(j, i, currFnum, iRange, QP, nRefFrames, FastME)
                # return [recons_block,mv, resE]
                # print(res)
                recT = res[0]
                mvT = res[1]
                resET = res[2]

                motion_V[i].append(mvT)
                QTC_F[i].append(resET)
                reconstructed_frame[i].append(recT)

        self.reconstructedFrame[currFnum] = self.constractFrame(reconstructed_frame)
        # print(self.VaribleBlockIndicators[currFnum])
        motionV = motion_V

        return [motionV, QTC_F]

    def generate_scaled_ref_frames(self, a_ref_frame):
        a_scaled_ref_frame = np.full((2 * self.heightV - 1, 2 * self.widthV - 1), 0)
        for i in range(2 * self.heightV - 1):
            if i % 2 == 0:
                for j in range(2 * self.widthV - 1):
                    if j % 2 == 0:
                        a_scaled_ref_frame[i][j] = a_ref_frame[i // 2][j // 2]
        for i in range(2 * self.heightV - 1):
            if i % 2 == 0:
                for j in range(2 * self.widthV - 1):
                    if j % 2 != 0:
                        a_scaled_ref_frame[i][j] = (a_scaled_ref_frame[i][j - 1] +
                                                    a_scaled_ref_frame[i][j + 1]) // 2
            else:
                for j in range(2 * self.widthV - 1):
                    if j % 2 == 0:
                        a_scaled_ref_frame[i][j] = (a_scaled_ref_frame[i - 1][j] +
                                                    a_scaled_ref_frame[i + 1][j]) // 2
        for i in range(2 * self.heightV - 1):
            if i % 2 != 0:
                for j in range(2 * self.widthV - 1):
                    if j % 2 != 0:
                        a_scaled_ref_frame[i][j] = (a_scaled_ref_frame[i][j - 1] +
                                                    a_scaled_ref_frame[i][j + 1] +
                                                    a_scaled_ref_frame[i - 1][j] +
                                                    a_scaled_ref_frame[i + 1][j]) // 4
        return a_scaled_ref_frame

    def generate_interpolated_reference_block(self, x, y, rx, ry, ref_f):
        '''
        according to rx and ry, generate the interpolated reference block
        :param x: the index of block in the horizontal direction
        :param y:
        :param rx:
        :param ry:
        :param ref_f: decoded frame
        :return:
        '''
        if rx % 2 == 0 and ry % 2 == 0:
            rx = rx // 2
            ry = ry // 2
            # same as not enabling fractional motion estimation
            ref_x = int(x * self.blockSize + rx)
            ref_y = int(y * self.blockSize + ry)
            refB = ref_f[ref_y:ref_y + self.blockSize,
                   ref_x:ref_x + self.blockSize]
        elif rx % 2 != 0 and ry % 2 == 0:
            # interpolate in x direction
            ref_x1 = int(x * self.blockSize + (rx - 1) // 2)
            ref_x2 = int(x * self.blockSize + (rx + 1) // 2)
            ref_y = int(y * self.blockSize + ry // 2)
            refB1 = ref_f[ref_y:ref_y + self.blockSize,
                    ref_x1:ref_x1 + self.blockSize]
            refB2 = ref_f[ref_y:ref_y + self.blockSize,
                    ref_x2:ref_x2 + self.blockSize]
            refB = (refB1 + refB2) / 2
        elif rx % 2 == 0 and ry % 2 != 0:
            # interpolate in y direction
            ref_y1 = int(y * self.blockSize + (ry - 1) // 2)
            ref_y2 = int(y * self.blockSize + (ry + 1) // 2)
            ref_x = int(x * self.blockSize + rx // 2)
            refB1 = ref_f[ref_y1:ref_y1 + self.blockSize,
                    ref_x:ref_x + self.blockSize]
            refB2 = ref_f[ref_y2:ref_y2 + self.blockSize,
                    ref_x:ref_x + self.blockSize]
            refB = (refB1 + refB2) / 2
        else:
            # interpolate in both directions
            ref_x1 = int(x * self.blockSize + (rx - 1) // 2)
            ref_x2 = int(x * self.blockSize + (rx + 1) // 2)
            ref_y1 = int(y * self.blockSize + (ry - 1) // 2)
            ref_y2 = int(y * self.blockSize + (ry + 1) // 2)
            B1 = ref_f[ref_y1:ref_y1 + self.blockSize,
                 ref_x1:ref_x1 + self.blockSize]
            B2 = ref_f[ref_y1:ref_y1 + self.blockSize,
                 ref_x2:ref_x2 + self.blockSize]
            B3 = ref_f[ref_y2:ref_y2 + self.blockSize,
                 ref_x1:ref_x1 + self.blockSize]
            B4 = ref_f[ref_y2:ref_y2 + self.blockSize,
                 ref_x2:ref_x2 + self.blockSize]
            refB1 = (B1 + B3) / 2
            refB2 = (B1 + B2) / 2
            refB3 = (B3 + B4) / 2
            refB4 = (B2 + B4) / 2
            refB = (refB1 + refB2 + refB3 + refB4) / 4

        return refB

    def diff_encode_intra_perVBS(self, per_frame, flag):
        encoded = []
        for row in range(len(per_frame)):
            firstBlockFlag = False
            previousUV = 0
            if flag[row][0] == 0:
                diff_row = [per_frame[row][0]]
            else:
                res = Diff_Intra_Decoder(per_frame[row][0])
                diff_row = res
            for col in range(1, len(per_frame[row])):
                if flag[row][col] == 1:
                    res = Diff_Intra_Decoder(per_frame[row][col])
                    diff_row = res
                else:
                    if firstBlockFlag is False:
                        diff_row.append(per_frame[row][col])
                        previousUV = col
                    else:
                        diff_row.append(per_frame[row][col] ^ per_frame[row][previousUV])
                        previousUV = col
            encoded.append(diff_row)
        return encoded

    def diff_encode_inter_perVBS(self, per_frame, flag):
        encoded = []
        for row in range(len(per_frame)):
            firstBlockFlag = False
            previousUV = 0
            if flag[row][0] == 0:
                diff_row = [per_frame[row][0]]
            else:
                res = Diff_Enco_inter(per_frame[row][0])
                diff_row = res
            for col in range(1, len(per_frame[row])):
                if flag[row][col] == 1:
                    res = Diff_Enco_inter(per_frame[row][col])
                    diff_row = res
                else:
                    if firstBlockFlag is False:
                        diff_row.append(per_frame[row][col])
                        previousUV = col
                    else:
                        diff_row.append((per_frame[row][previousUV][0] - per_frame[row][col][0],
                                         per_frame[row][previousUV][1] - per_frame[row][col][1],
                                         per_frame[row][col][2]))
                        previousUV = col
            encoded.append(diff_row)
        return encoded

    def visualizeVBS(self, frame_num):
        image = self.reconstructedFrame[frame_num]

        # Define block size
        block_size = self.blockSize

        # Randomly generate flags for each block
        flags_matrix = np.array(self.VaribleBlockIndicators[frame_num])

        # Draw vertical grid lines
        for i in range(1, image.shape[1] // block_size):
            cv2.line(image, (i * block_size, 0), (i * block_size, image.shape[0]), color=0, thickness=1)

        # Draw horizontal grid lines
        for j in range(1, image.shape[0] // block_size):
            cv2.line(image, (0, j * block_size), (image.shape[1], j * block_size), color=0, thickness=1)

        # Draw "十" in the blocks where the flag is 1
        for j in range(flags_matrix.shape[0]):
            for i in range(flags_matrix.shape[1]):
                if flags_matrix[j][i] == 1:
                    # Draw a cross ("十")
                    # cv2.putText(image, str(3),
                    #             (i * block_size + block_size // 2, j * block_size + block_size // 2),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, color=0, thickness=1)
                    cv2.line(image, (i * block_size + block_size // 2, j * block_size),
                             (i * block_size + block_size // 2, (j + 1) * block_size), color=0, thickness=1)
                    cv2.line(image, (i * block_size, j * block_size + block_size // 2),
                             ((i + 1) * block_size, j * block_size + block_size // 2), color=0, thickness=1)

        # Display the image with grids and crosses
        cv2.imshow("Image with Grids and Crosses", image)
        cv2.imwrite('VBS.jpg', image)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualizeMV(self, frame_num):
        image = self.reconstructedFrame[frame_num]
        print(len(self.MV))
        mv = self.MV[frame_num - math.ceil(frame_num / len(self.I_p))]
        # Define block size
        block_size = self.blockSize

        # Randomly generate flags for each block
        flags_matrix = np.array(self.VaribleBlockIndicators[frame_num])

        # Draw vertical grid lines
        for i in range(1, image.shape[1] // block_size):
            cv2.line(image, (i * block_size, 0), (i * block_size, image.shape[0]), color=0, thickness=1)

        # Draw horizontal grid lines
        for j in range(1, image.shape[0] // block_size):
            cv2.line(image, (0, j * block_size), (image.shape[1], j * block_size), color=0, thickness=1)

        # Draw "十" in the blocks where the flag is 1
        for j in range(flags_matrix.shape[0]):
            for i in range(flags_matrix.shape[1]):
                if flags_matrix[j][i] == 0:
                    arrow_length = 5
                    dx, dy = mv[j][i][0] * arrow_length, mv[j][i][1] * arrow_length
                    cv2.arrowedLine(image,
                                    (i * block_size + block_size // 2, j * block_size + block_size // 2),
                                    (int(i * block_size + block_size // 2 + dx),
                                     int(j * block_size + block_size // 2 + dy)),
                                    color=0, thickness=1, tipLength=0.2)
                    # Draw a cross ("十")
                    # cv2.putText(image, str(mv[j][i][2]),
                    #             (i * block_size + block_size // 2, j * block_size + block_size // 2),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, color=0, thickness=1)
                    # cv2.line(image, (i * block_size + block_size // 2, j * block_size),
                    #          (i * block_size + block_size // 2, (j + 1) * block_size), color=0, thickness=1)
                    # cv2.line(image, (i * block_size, j * block_size + block_size // 2),
                    #          ((i + 1) * block_size, j * block_size + block_size // 2), color=0, thickness=1)

        # Display the image with grids and crosses
        cv2.imshow("Image with Grids and Crosses", image)
        cv2.imwrite('VBS.jpg', image)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

    def encoderMP(self, blockSize=16, searchRange=4, QP=6, FMEEnable=1, VBSEnable=1, FastME=1, nReferenceframe=1,
                  paraMode=1):
        self.QP = QP
        blackframe = np.full((self.heightV, self.widthV), 128, np.uint8)
        self.VBSEnable = VBSEnable
        self.reconstructedFrame = [[] for _ in range(self.frameNum)]
        self.MV = []
        self.QTCC = []
        self.MODE = []
        self.VaribleBlockIndicators = []
        self.blockSpliting(blockSize)
        # initialize the FMEEnable parameter and the scaled frames
        self.FMEEnable = FMEEnable
        self.scaled_frames = []
        self.bitsize = []
        bitsize = 0
        for i in range(10):  # !!!!!!!!!!!!!!!!!!!!!
            # for i in range(1):
            if paraMode == 1:
                f = self.blockedYF[i]
                motion_V, QTC_F, reconstructed_frame = searchT1(i, searchRange, f, blockSize, QP)
                self.MV.append(motion_V)
                self.QTCC.append(QTC_F)
                self.reconstructedFrame[i] = reconstructed_frame
                bitsize += 8 * 2 * self.blockNumInHeight * self.blockNumInWidth
                for y in QTC_F:
                    for x in y:
                        bitsize += len(x) * 8
                print("bitsize", bitsize)
                self.bitsize.append(bitsize)
            if paraMode == 2:
                f = self.blockedYF[i]
                if self.iPer[i] == 0:
                    # inter para
                    if i == 0:
                        motion_V, QTC_F, reconstructed_frame = Full_searchparaT2(i, searchRange, f, blackframe,
                                                                                 blockSize, QP)
                    else:
                        motion_V, QTC_F, reconstructed_frame = Full_searchparaT2(i, searchRange, f,
                                                                                 self.reconstructedFrame[i - 1],
                                                                                 blockSize, QP)
                    self.MV.append(motion_V)
                    self.QTCC.append(QTC_F)
                    self.reconstructedFrame[i] = self.constractFrame(reconstructed_frame)
                    bitsize += 8 * 2 * self.blockNumInHeight * self.blockNumInWidth
                    a = count_elements_in_nested_array(QTC_F)
                    bitsize += a * 8
                    print("bitsize", bitsize)
                    self.bitsize.append(bitsize)

                if self.iPer[i] == 1:
                    # intra para
                    mode, QTC_F, re_frame, VaribleBlockFlag = intra_Pred_T2(f, i, blockSize, QP)
                    self.MODE.append(mode)
                    self.QTCC.append(QTC_F)
                    self.reconstructedFrame[i] = self.constractFrame(re_frame)
                    a = count_elements_in_nested_array(QTC_F)

                    bitsize += a * 8
                    m = count_elements_in_nested_array(mode)
                    bitsize += m * 1
                    print("bitsize", bitsize)
                    self.bitsize.append(bitsize)
            if paraMode == 3:
                break
        if paraMode == 3:

            for i in range(1):
                # intra para
                f = self.blockedYF[0]
                mode, QTC_F, re_frame, VaribleBlockFlag = intra_Pred_T2(f, i, blockSize, QP)
                self.MODE.append(mode)
                self.QTCC.append(QTC_F)
                self.reconstructedFrame[i] = self.constractFrame(re_frame)
                a = count_elements_in_nested_array(QTC_F)

                bitsize += a * 8
                m = count_elements_in_nested_array(mode)
                bitsize += m * 1
                print("bitsize", bitsize)
                for i in range(1, 10, 2):
                    ref_f = np.array(self.reconstructedFrame[i - 1])
                    print(ref_f.shape)
                    f = self.blockedYF[i]
                    f1 = self.blockedYF[i + 1]

                    # def T3main(currFnum, iRange, currF1, currF2, refF, blockSize, QP=6, heightV=288, widthV=352):
                    r1, r2 = T3main((i, i + 1), 4, f, f1, ref_f, blockSize, 6, 288, 352)

                    # return motion_V, QTC_F, reconstructed_frame
                    self.MV.append(r1[0])
                    self.MV.append(r2[0])
                    self.QTCC.append(r1[1])
                    self.QTCC.append(r2[1])

                    self.reconstructedFrame[i] = self.constractFrame((r1[2]))
                    self.reconstructedFrame[i + 1] = self.constractFrame((r2[2]))
                    bitsize += 8 * 2 * self.blockNumInHeight * self.blockNumInWidth * 2
                    a = count_elements_in_nested_array(r1[1])
                    a1 = count_elements_in_nested_array(r2[1])
                    bitsize += (a + a1) * 8
                    print("bitsize", bitsize)

        # Decode MVs
        if VBSEnable == 0:
            diff_inter_lis = []
            for i in self.MV:
                # print(i)
                diff_inter = Diff_Enco_inter(i)
                # print(diff_inter)
                diff_inter_lis.append(diff_inter)
            diff_intra_lis = []
            for i in self.MODE:
                diff_intra = Diff_Intra_Encoder(i)
                diff_intra_lis.append(diff_intra)
        else:
            indI = 0
            indP = 0
            diff_inter_lis = []
            diff_intra_lis = []
            for i in range(len(self.VaribleBlockIndicators)):
                if self.iPer[i] == 0:
                    res = self.diff_encode_inter_perVBS(self.MV[indP], self.VaribleBlockIndicators[i])
                    diff_inter_lis.append(res)
                    indP += 1
                if self.iPer[i] == 1:
                    res = self.diff_encode_intra_perVBS(self.MODE[indI], self.VaribleBlockIndicators[i])
                    diff_intra_lis.append(res)
                    indI += 1

        QTCCoeeff = [self.QTCC]
        MDiff = [self.iPer, diff_inter_lis, diff_intra_lis, self.VaribleBlockIndicators]  # self.VaribleBlockIndicators
        return [QTCCoeeff, MDiff]

    def visualizeMTF(self, frame_num):
        image = self.reconstructedFrame[frame_num]
        print(len(self.MV))
        mv = self.MV[frame_num - math.ceil(frame_num / len(self.I_p))]
        # Define block size
        block_size = self.blockSize

        # Randomly generate flags for each block
        flags_matrix = np.array(self.VaribleBlockIndicators[frame_num])

        # Draw vertical grid lines
        for i in range(1, image.shape[1] // block_size):
            cv2.line(image, (i * block_size, 0), (i * block_size, image.shape[0]), color=0, thickness=1)

        # Draw horizontal grid lines
        for j in range(1, image.shape[0] // block_size):
            cv2.line(image, (0, j * block_size), (image.shape[1], j * block_size), color=0, thickness=1)

        # Draw "十" in the blocks where the flag is 1
        for j in range(flags_matrix.shape[0]):
            for i in range(flags_matrix.shape[1]):
                if flags_matrix[j][i] == 0:
                    # Draw a cross ("十")
                    cv2.putText(image, str(mv[j][i][2]),
                                (i * block_size + block_size // 2, j * block_size + block_size // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, color=0, thickness=1)
                    # cv2.line(image, (i * block_size + block_size // 2, j * block_size),
                    #          (i * block_size + block_size // 2, (j + 1) * block_size), color=0, thickness=1)
                    # cv2.line(image, (i * block_size, j * block_size + block_size // 2),
                    #          ((i + 1) * block_size, j * block_size + block_size // 2), color=0, thickness=1)

        # Display the image with grids and crosses
        cv2.imshow("Image with Grids and Crosses", image)
        cv2.imwrite('VBS.jpg', image)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()


'''
EXE4 a
'''


def dct_2d(block):
    # Apply 2D DCT to the input block
    dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    # dct_block = np.round(dct_block)
    return dct_block


def idct_2d(dct_block):
    # Apply 2D inverse DCT to the DCT coefficients
    idct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
    # round to the nearest integer
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


# for a frame
def Diff_Enco_inter(motion_v):
    encoded = []
    for sequence in motion_v:
        diff = [sequence[0]]
        for i in range(1, len(sequence)):
            diff.append((sequence[i][0] - sequence[i - 1][0], sequence[i][1] - sequence[i - 1][1], sequence[i][2]))
        encoded.append(diff)
    return encoded


# for a frame
def Diff_Deco_inter(encoded):
    decoded = []
    for sequence in encoded:
        recons = [sequence[0]]
        for i in range(1, len(sequence)):
            recons.append((recons[i - 1][0] + sequence[i][0], recons[i - 1][1] + sequence[i][1], sequence[i][2]))
        decoded.append(recons)
    return decoded


'''
EXE4 c
'''


def Diff_Intra_Encoder(matrix):
    encoded = []
    for row in range(len(matrix)):
        diff_row = [matrix[row][0]]
        for col in range(1, len(matrix[row])):
            diff_row.append(matrix[row][col] ^ matrix[row][col - 1])
        encoded.append(diff_row)
    return encoded


# for a frame
def Diff_Intra_Decoder(encoded):
    decoded = []
    for row in range(len(encoded)):
        recons_row = [encoded[row][0]]
        for col in range(1, len(encoded[row])):
            recons_row.append(recons_row[col - 1] ^ encoded[row][col])
        decoded.append(recons_row)
    return decoded
    # D


'''
EXE4 d
'''


def entropy_encoder(matrix):
    reorderM = reorder(matrix)
    res = rle_encode(reorderM)
    return res


def decode_entropy(n, encoded):
    co = rle_decode(encoded, n)
    res = reorder_decoder(co)
    return np.array(res)


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


def rle_decode(arr, lenT):
    decoded = []
    i = 0

    while i < len(arr):
        if arr[i] < 0:  # Non-zero terms
            count_non_zero = abs(arr[i])
            decoded.extend(arr[i + 1: i + count_non_zero + 1])
            i += count_non_zero + 1
        elif arr[i] == 0:  # End of non-zero terms
            break
        else:  # Zero terms
            count_zero = arr[i]
            decoded.extend([0] * count_zero)
            i += 1
    if len(decoded) < lenT:
        zeros = lenT - len(decoded)
        for i in range(zeros):
            decoded.append(0)
    return decoded


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


def reorder_decoder(coefficients):
    size = int(len(coefficients) ** 0.5)
    reordered_matrix = [[0 for _ in range(size)] for _ in range(size)]
    index = 0
    for i in range(size + size - 1):
        if i % 2 == 0:  # Even diagonals
            row = min(i, size - 1)
            col = max(0, i - size + 1)
            while row >= 0 and col < size:
                reordered_matrix[row][col] = coefficients[index]
                index += 1
                row -= 1
                col += 1
        else:  # Odd diagonals
            col = min(i, size - 1)
            row = max(0, i - size + 1)
            while col >= 0 and row < size:
                reordered_matrix[row][col] = coefficients[index]
                index += 1
                col -= 1
                row += 1
    return reordered_matrix


'''
Exe4 d
'''


def Ex_Go_enco(num):
    if num <= 0:
        trans_num = -2 * num + 1
    else:
        trans_num = 2 * num
    num_bits = trans_num.bit_length()
    unary_code = "0" * (num_bits - 1)
    binary_code = bin(trans_num)[2:]
    exp_golomb_code = unary_code + binary_code
    return exp_golomb_code


def decode_exp_golomb(exp_golomb_code):
    l = len(exp_golomb_code)
    unary_length = (l - 1) // 2
    binary_part = exp_golomb_code[unary_length:]
    trans_num = int(binary_part, 2)
    print(trans_num)
    if trans_num % 2 == 0:
        num = trans_num // 2
    else:
        num = -((trans_num - 1) // 2)
    return num


def psnr(original, reconstructed, max_val=255):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * np.log10((max_val ** 2) / mse)
    return psnr_value


def mse(original, reconstructed, max_val=255):
    mse = np.mean((original - reconstructed) ** 2)
    return mse


def EncoderOutVBSOff(res):
    '''
    save the decoded QTCC and MDIFF into files
    :param res:
    :return:
    '''
    QTCC = res[0][0]

    res1 = []
    for fr in QTCC:
        for y in fr:
            for x in y:
                res1 = res1 + x
    arr = np.array(res1)
    a_r = arr.astype(dtype=np.uint8)
    np.save('QTCC', a_r)

    MDiff = res[1]
    res2 = []
    for i in MDiff:
        lis = np.array([i])
        lis = lis.flatten()
        lisPA = lis.tolist()
        res2 = lisPA + res2

    arr = np.array(res2)
    a_r = arr.astype(dtype=np.int8)
    np.save('MDiff', a_r)


def EncoderOut(res, ip):
    '''
    save the decoded QTCC and MDIFF into files
    :param res:
    :return:
    '''
    QTCC = res[0][0]
    flag = res[1][3]
    res1 = []
    frn = 0
    for fr in QTCC:
        yi = 0
        for y in fr:
            xi = 0
            for x in y:
                # print(flag[frn][yi][xi])
                if flag[frn][yi][xi] == 0:
                    # res1 = res1 + x
                    res1 = np.concatenate((res1, x))
                else:
                    for y2 in range(len(x)):
                        for x2 in range(len(x[0])):
                            res1 = np.concatenate((res1, x[y2][x2]))
                xi += 1
            yi += 1
        frn += 1
        # res1 = res1 + x
        # res1 = np.concatenate((res1, x))

    # print(len(res1))
    arr = np.array(res1)
    a_r = arr.astype(dtype=np.int8)
    np.save('QTCC', a_r)

    MDiff = res[1]

    # QTCCoeeff = [self.QTCC]
    # MDiff = [self.iPer, diff_inter_lis, diff_intra_lis, self.VaribleBlockIndicators]  # self.VaribleBlockIndicators
    inter = MDiff[1]
    intra = MDiff[2]

    # print(inter)
    res1 = []
    indP = 0
    count = 0
    for fr in inter:
        if indP == 0:
            frn = 1
        else:
            frn = indP + math.ceil(frn / ip)
        for iy in range(len(fr)):
            for ix in range(len(fr[0])):
                # print("F:",flag[frn][iy][ix])
                # print("V:", fr[iy][ix])

                if len(fr[iy][ix]) == 3:
                    count += 1
                    res1 = np.concatenate((res1, fr[iy][ix]))
                else:
                    for y2 in range(2):
                        for x2 in range(2):
                            count += 1
                            res1 = np.concatenate((res1, fr[iy][ix][y2][x2]))
        indP += 1
    print(len(res1))
    a_r = res1.astype(dtype=np.int8)
    np.save('inter', a_r)

    res1 = []
    indI = 0
    count = 0
    for fr in intra:
        if indI == 0:
            frn = 0
        else:
            frn = indI * ip
        for iy in range(len(fr)):
            for ix in range(len(fr[0])):
                print("F:", flag[frn][iy][ix])
                print("V:", fr[iy][ix])

                if fr[iy][ix] == 1 or fr[iy][ix] == 0:
                    count += 1
                    res1.append(fr[iy][ix])
                else:
                    for y2 in range(2):
                        for x2 in range(2):
                            count += 1
                            res1.append(fr[iy][ix][y2][x2])
        indI += 1
    # print(res1)
    res1 = np.array(res1)
    a_r = res1.astype(dtype=np.int8)
    np.save('intraDIff', a_r)

    res1 = np.array(flag)
    a_r = res1.astype(dtype=np.int8)
    np.save('Flag', a_r)


def EncoderOut2(res):
    QTCC = res[0]

    res1 = []
    # for fr in QTCC:
    for y in QTCC:
        for x in y:
            res1 = res1 + x
    arr = np.array(res1)
    a_r = arr.astype(dtype=np.uint8)
    np.save('QTCC', a_r)

    MDiff = res[1]
    res2 = []
    for i in MDiff:
        lis = np.array([i])
        lis = lis.flatten()
        lisPA = lis.tolist()

        res2 = lisPA + res2

    arr = np.array(res2)
    a_r = arr.astype(dtype=np.int8)
    np.save('MDiff', a_r)


def RDFrame(F1, F2, bitsize):
    res = []
    lam = (0.04) * 2 ** ((QP - 12) / 3)

    D = mse(F1, F2)
    print("MSE: ", D)
    J = D + lam * bitsize
    res.append(J)
    return res


def count_elements_in_nested_array(arr):
    count = 0
    for item in arr:
        if isinstance(item, list):
            count += count_elements_in_nested_array(item)
        else:
            count += 1
    return count


if __name__ == "__main__":
    '''
    The main function will encode and decode the first 10 frame video 
    '''
    newO = VideoEncoder('foremanakiyo.yuv', 2, 'IPPPPPPPPPPPPPPPPPPPP')
    QPs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    QP = 9

    # start_time = time.time()
    # res = newO.encoder(16, 16, QP, 1, 0, 0, 1)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"execution time: {execution_time} s")
    # res = newO.RC_encoder(RCflag=1,targetBR = 2737152,fps = 30,blockSize=16, searchRange=16, QP=QP, QPs= QPs,FMEEnable=1, VBSEnable=0, FastME=0, nReferenceframe=1)
    starttime = time.time()
    res = newO.RC_encoder(RCflag=3, targetBR=360000, fps=30, blockSize=16, searchRange=16, QP=QP, QPs=QPs, FMEEnable=1,
                          VBSEnable=0, FastME=0, nReferenceframe=1)
    endtime = time.time()
    duration = endtime - starttime
    print(duration)
    # print( newO.bit_table(16,QPs))
    # print( newO.bit_table(16,QPs))
    # res = newO.RC_encoder(RCflag=1,targetBR = 1094861,fps = 30,blockSize=16, searchRange=16, QP=QP, QPs= QPs,FMEEnable=1, VBSEnable=0, FastME=0, nReferenceframe=1)

    # print(newO.row_bitcount_proportion)

    # print(newO.bit_count_row(16,11))
    # newO.decoder(res[0], res[1])
    bit_table = {0: 8139, 1: 6638, 2: 5101, 3: 3635, 4: 2218, 5: 1143, 6: 525, 7: 287, 8: 223, 9: 207}
    bit_table_16 = {0: 39930, 1: 32548, 2: 24981, 3: 17838, 4: 10870, 5: 5557, 6: 2516, 7: 1353, 8: 1063, 9: 1004,
                    10: 995, 11: 994}
    '''
        calculate the size when VBS is on
    '''
    # D_sum = 0
    # for i in range(10):
    #     current_frame = newO.yFrame[i]
    #     reconstructed_frame = newO.reconstructedFrame[i]
    #     D_sum += mse(current_frame, reconstructed_frame)
    #
    # EncoderOut(res, 7)
    # # EncoderOutVBSOff(res)
    #
    # if os.path.exists('QTCC.npy') and \
    #         os.path.exists('inter.npy') and \
    #         os.path.exists('intraDIff.npy') and \
    #         os.path.exists('Flag.npy'):
    #     R_sum = os.path.getsize('QTCC.npy') + \
    #             os.path.getsize('inter.npy') + \
    #             os.path.getsize('intraDIff.npy') + \
    #             os.path.getsize('Flag.npy')
    #     R_sum *= 8
    # # if os.path.exists('QTCC.npy') and os.path.exists('MDiff.npy'):
    # #     R_sum = os.path.getsize('QTCC.npy') + os.path.getsize('MDiff.npy')
    # #     R_sum *= 8
    # lam = 0.2 * 2 ** ((QP - 12) / 3)
    # print(D_sum)
    # print(R_sum)
    # RD_cost = (D_sum + lam * R_sum) / 10
    # print(RD_cost)

    '''
        calculate the per-frame distortion and bitcount
    '''
    # iNum = 0
    # pNum = 0
    # row_bit = []
    # total = 0
    # for i in range(21):
    #     if i % 21 != 0:
    #         pNum += 1
    #         QTCC = res[0][0][i]
    #         MDiff1 = res[1][0][i]
    #         MDiff2 = res[1][1][pNum - 1]
    #         MDiff3 = []
    #     else:
    #         QTCC = res[0][0][i]
    #         MDiff1 = res[1][0][i]
    #         MDiff2 = []
    #         MDiff3 = res[1][2][iNum - 1]
    #     MDiff = [MDiff1, MDiff2, MDiff3]
    #     output = [QTCC, MDiff]
    #     EncoderOut2(output)
    #
    #     if os.path.exists('QTCC.npy') and os.path.exists('MDiff.npy') :
    #         frame_bytescount = os.path.getsize('QTCC.npy') + os.path.getsize('MDiff.npy')
    #     frame_bitscount = frame_bytescount * 8
    #     #row_bitcount = frame_bitscount/88
    #     #row_bit.append(row_bitcount)
    # #filtered_list = [value for index, value in enumerate(row_bit) if index % 8 != 0]
    # #print(round(sum(filtered_list) / len(filtered_list)))
    #     #print('F:',row_bitcount)
    #     print('F:',frame_bitscount)
    #     total += frame_bitscount
    # print(total)
    # print(''' ''')
    # for i in range(21):
    #      current_frame = newO.yFrame[i]
    #      reconstructed_frame = newO.reconstructedFrame[i]
    #      print(psnr(current_frame, reconstructed_frame))

    '''
        calculate the percentage of split blocks
    '''
    # avg_percentage = 0
    # block_num = newO.blockNumInHeight * newO.blockNumInWidth
    # for i in range(10):
    #     fr = newO.VaribleBlockIndicators[i]
    #     avg_percentage += np.sum(fr)/block_num/10
    # print(avg_percentage)

    '''
        calculate the size
    '''
    D_sum = 0
    for i in range(21):
        current_frame = newO.yFrame[i]
        reconstructed_frame = newO.reconstructedFrame[i]
        D_sum += mse(current_frame, reconstructed_frame)

    EncoderOutVBSOff(res)

    if os.path.exists('QTCC.npy') and os.path.exists('MDiff.npy'):
        R_sum = os.path.getsize('QTCC.npy') + os.path.getsize('MDiff.npy')
        R_sum *= 8
    lam = 0.02 * 2 ** ((QP - 12) / 3)
    print(D_sum)
    print(R_sum)
    RD_cost = (D_sum + lam * R_sum) / 21
    print(RD_cost)

    '''
        visualization
    '''
    # newO.visualizeVBS(0)
    # # print(res[1][1])
    # # EncoderOut(res) # This line output the encoded values into files
    # newO.decoder(res[0], res[1])

    # newO.blockSpliting(16)
    # newO.reconstructedFrame = [1]
    # newO.VaribleBlockIndicators = []
    # newO.FMEEnable = 1
    # #
    # r = newO.intra_Pred_V(0, 6)
    # newO.QP = 6
    # # print(r[1])
    # newO.intra_V_decoder(0, r[0], r[1])
    # # # print(r[0])
    # # newO.visualizeVBS(0)
    # # # newO.diff_encode_intra_perVBS(r[0], newO.VaribleBlockIndicators[0])
