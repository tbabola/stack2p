from importlib.metadata import files
import xml.etree.ElementTree as ET
import math
import numpy as np
import suite2p
import time


def multisamplingAverage(bin, samplesPerPixel):
    bin = bin.reshape(-1,samplesPerPixel)
    addmask = np.sum(bin <= 2**13,1,dtype="uint16")
    bin[bin > 2**13] = 0
    bin = np.floor_divide(np.sum(bin,1,dtype="uint16"),addmask)

    return bin

def parse_bruker_xml(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    b_xml = {}
    for thing in root.findall("./PVStateShard/"):
        try:
            b_xml[thing.attrib['key']] = thing.attrib['value']
        except:
            pass

    for thing in root.findall("Sequence"):
        try:
            print(thing.attrib['key']) 
            print(thing.attrib['value'])
        except:
            pass
    
    #if this is a time series with z-stack
    if root.findall("Sequence")[0].attrib['type'] == 'TSeries ZSeries Element':
        b_xml['bidirectional'] = root.findall("Sequence")[0].attrib['bidirectionalZ']
        b_xml['nplanes'] = len(root.findall("Sequence")[0]) - 1
    elif root.findall("Sequence")[0].attrib['type'] == 'TSeries Timed Element':
        b_xml['nplanes'] = 1
    
    b_xml['ncycles'] = len(root) - 2
    b_xml['nframes'] = [len(root[i+2].findall('Frame')) for i in range(b_xml['ncycles'])] #not very pythonic, but just counts frames per cycle
    b_xml['nchannels'] = len(root[2].findall('Frame')[0].findall('File')) 
    return b_xml

def find_bruker_raw_files(ops_m, ops1, folder_num):
    """  finds bruker raw files
        Parameters
        ----------
        ops1 : list of dictionaries
        'keep_movie_raw', 'data_path', 'look_one_level_down', 'reg_file'...

        Returns
        -------
            ops1 : list of dictionaries
                adds fields 'filelist', 'first_tiffs', opens binaries

    """
    fs = list(Path(ops_m['data_path'][folder_num]).glob("*RAWDATA*"))
    fs = [str(file) for file in fs]
    for ops_s in ops1:
        if folder_num == 0:
            ops_s['filelist'] = fs
        else:
            ops_s['filelist'].extend(fs)

    return ops1, fs 

def filesToFrames(datafiles, brukerXml, ops):
    filesToFrames = []
    max_file_size = 2048000000 #in bytes
    bytes_per_pixel = 2
    samples_per_pixel = int(brukerXml['samplesPerPixel'])
    pixelsPerLine = ops['Lx']
    LinesPerFrame = ops['Ly']
    bytesPerFrame = bytes_per_pixel*samples_per_pixel*pixelsPerLine*LinesPerFrame
    #brukerXml[]
    counter = 0
    for i in range(ops['nframes']):
        actualStartByte = i*bytes_per_pixel*samples_per_pixel*pixelsPerLine*LinesPerFrame
        startByte = actualStartByte % max_file_size
        endByte = startByte + bytesPerFrame
        fileNum = math.floor(actualStartByte/max_file_size)
        filesToFrames.append({'file':datafiles[fileNum],'startByte':startByte,'endByte':endByte})
    
    return filesToFrames

def filesToIndividualFrames(datafiles, brukerXml, ops, startFrame, endFrame):
    #currently only working with 1 frames, have not gone to two frames
    filesToFrames = []
    max_file_size = 2097152000 #in bytes
    bytes_per_pixel = 2
    samples_per_pixel = int(brukerXml['samplesPerPixel'])
    pixelsPerLine = ops['Lx']
    LinesPerFrame = ops['Ly']
    bytesPerFrame = bytes_per_pixel*samples_per_pixel*pixelsPerLine*LinesPerFrame
    #brukerXml[]
    counter = 0
    
    actualStartByte = startFrame*bytes_per_pixel*samples_per_pixel*pixelsPerLine*LinesPerFrame
    actualEndByte = endFrame*bytes_per_pixel*samples_per_pixel*pixelsPerLine*LinesPerFrame + bytesPerFrame
    startFile = actualStartByte // max_file_size
    endFile = actualEndByte // max_file_size
    startByte = actualStartByte % max_file_size
    endByte = actualEndByte % max_file_size

    if startFile == endFile:
        tempCount = int((endByte - startByte) / bytes_per_pixel)
        filesToFrames.append({'file':datafiles[startFile],'startByte':startByte,'count':tempCount})
    else:
        for i in range(startFile,endFile+1,1):
            if i == startFile:
                filesToFrames.append({'file':datafiles[i],'startByte':startByte,'count':-1})
            elif i == endFile:
                tempCount = int(endByte / bytes_per_pixel)
                if tempCount:
                    filesToFrames.append({'file':datafiles[i],'startByte':0,'count':tempCount})
            else:
                filesToFrames.append({'file':datafiles[i],'startByte':0,'count':-1}) 

    bytes = []
    for data in filesToFrames:
        bytes.append(np.fromfile(data['file'], dtype = np.uint16, offset=data['startByte'],count=data['count']))
    return np.concatenate(bytes)

def register_frames(frames, ops, startFrame, endFrame):
    startTime = time.time()
    blocks = suite2p.registration.nonrigid.make_blocks(ops['Ly'],ops['Lx'],ops['block_size'])
    print("Blocks made", time.time()-startTime)
    reg_frames = suite2p.registration.register.shift_frames(
        frames,
        ops['yoff'][startFrame:endFrame+1],
        ops['xoff'][startFrame:endFrame+1],
        ops['yoff1'][startFrame:endFrame+1],
        ops['xoff1'][startFrame:endFrame+1],
        blocks=blocks,
        ops=ops)
    print("reg command", time.time()-startTime)
    return reg_frames

def convertBytesToFrames(bytes, bruker_xml):
    samplesPerPixel = int(bruker_xml['samplesPerPixel'])
    nXpixels = int(bruker_xml['pixelsPerLine'])
    nYpixels = int(bruker_xml['linesPerFrame'])
    nchannels = bruker_xml['nchannels']

    bin = bytes - 2**13 #weird quirk when capturing with galvo-res scanner

    if samplesPerPixel > 1:
        bin = multisamplingAverage(bin,samplesPerPixel)
    elif samplesPerPixel == 1:
        bin[bin > 2**13] = 0

    frames = []
    #currently only working with 1 frames, have not gone to two frames in filesToIndiviudalFrames function
    for chan in range(nchannels):
        #grab appropriate samples and flip every other line
        bin_temp = bin[chan::nchannels]
        bin_temp = bin_temp.reshape(-1,nXpixels,nYpixels)
        bin_temp[:,1::2,:] = np.flip(bin_temp[:,1::2,:],2)
        frames.append(bin_temp)
    
    return frames[0]

def test():
    print("test passed")