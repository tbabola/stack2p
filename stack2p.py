import suite2p
from pathlib import Path
import mat73
import numpy as np
import utils
import time

class stack2p():
    def __init__(self, directory):
        self.directory = directory
        self.soundData = None
        self.ops = None
        self.framesToBytes = None
        self.brukerXML = None
        self.dataFiles = None
        
        sound_file_dir = list(directory.glob("sound_file*"))
        if not sound_file_dir:
            print("Warning: No sound file detected.")
        else:
            self.soundData = mat73.loadmat(sound_file_dir[0])

        ops_file = list(directory.glob("suite2p\plane0\ops.npy"))
        if not ops_file:
            print("Warning: No ops file detected.")
        else:
            self.ops = np.load(ops_file[0],allow_pickle=True)[()]

        bruker_xml_file = list(directory.glob("*.xml"))
        if not bruker_xml_file:
            print("Warning: No Bruker XML file detected.")
        else:
            self.brukerXML = utils.parse_bruker_xml(bruker_xml_file[0])

        dataFiles = list(directory.glob("*RAWDATA*"))
        if not dataFiles:
            print("Warning: No raw data files found.")
        else:
            self.dataFiles = dataFiles

    def generateRegBinary(self, batch=500):
        savePath = self.directory / "suite2p/plane0/data_rereg.bin"
        reg_file = open(savePath, 'wb')
        count = 0
        startTime = time.time()
        while count < self.ops['nframes']:
            if self.ops['nframes'] - count < batch:
                batch = self.ops['nframes'] - count
            frames = self.getFrames(count,count+batch-1)
            reg_file.write(bytearray(frames))
            count = count + batch
            print("Iteration",count,time.time()-startTime)

    def getFrames(self, start_frame, end_frame):
        start_time = time.time()
        framesToBytes = utils.filesToIndividualFrames(self.dataFiles,self.brukerXML,self.ops,start_frame,end_frame)
        print("frames shape",framesToBytes.shape)
        frames = utils.convertBytesToFrames(framesToBytes,self.brukerXML)
        print("Time to load frames:", time.time()-start_time)
        start_time = time.time()
        reg_frames = utils.register_frames(np.int16(frames),self.ops,start_frame,end_frame)
        print("Time to reg frames:", time.time() - start_time)

        return reg_frames.astype(np.int16)

    



if __name__ == "__main__":
    tp = Path("X:\\Travis\\Cdh23 Data\\m984\\2P\\L4\\220812")

    # print(dir(utils))
    # print()
    test = stack2p(tp)
    test.generateRegBinary(batch=500)
    startTime = time.time()

    path2 = tp / "suite2p/plane0/data.bin"
    data2 = np.fromfile(path2, dtype=np.uint16).reshape(-1, 1024, 1024)

    path3 = tp / "suite2p/plane0/data_rereg.bin"
    data3 = np.fromfile(path2, dtype=np.uint16).reshape(-1, 1024, 1024)

    print(np.array_equal(data2,data3))
    # for i in range(0,750,50):
    #     test.getFrames(i,i+100-1)
    #     print("Time to register (total): ",time.time()-startTime,"\n\n")
    # data1 = test.frames
    # data1_reg = test.reg_frames
    #
    # path2 = Path("Z:\\tbabola\\Experiments\\2021\\211203_QuickMap\\QuickMap-007 - Copy\\suite2p\\plane0\\data_raw.bin")
    # data2 = np.fromfile(path2,dtype=np.uint16, count=1024*1024*10).reshape(-1,1024,1024)
    #
    # path3 = Path(
    #     "Z:\\tbabola\\Experiments\\2021\\211203_QuickMap\\QuickMap-007 - Copy\\suite2p\\plane0\\data.bin")
    # data3 = np.fromfile(path3, dtype=np.uint16, count=1024 * 1024 * 10).reshape(-1, 1024, 1024)

    #test_ops_path = Path("X:\\Travis\\Cdh23 Data\\m985\\2P\\L23\\220812_1\\")
    #test2 = stack2p(test_ops_path)
    #test_ops_path = Path("X:\\Travis\\Cdh23 Data\\m985\\2P\\L23\\220812_2\\")
    #test3 = stack2p(test_ops_path)

    # import matplotlib.pyplot as plt
    # plt.hist(data2[0,:,:]-data1[0,:,:])
    # plt.show()
    #
    # plt.imshow(data2[0, :, :] - data1[0, :, :])
    # plt.show()

    # plt.imshow(data1[0,:,:])
    # plt.show()
    # print(data1.shape,data2.shape, data1.dtype,data2.dtype,)
    # print(np.array_equal(data1[0:10,:,:],data2[0:10,:,:]))
    # print(np.array_equal(data1_reg[0:10, :, :].as_type(np.int16), data3[0:10, :, :]))
    #


