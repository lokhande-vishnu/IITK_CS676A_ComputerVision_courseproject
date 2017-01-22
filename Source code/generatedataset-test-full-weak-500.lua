require 'cunxn'
foo=nxn.Dataset()



imglist=torch.DiskFile('trainval0.2.txt', 'r')

--vocfolder='/data/gpuuser1/voc12-test/VOC2012/JPEGImages/'
vocfolder='/data/gpuuser1/voc12-trainval/VOC2012/JPEGImages/'
foo:setTargetDir('../datasets/test_2308_4')
numimages=2308
batchsize=4

filelist={}
for i=1,numimages do
   table.insert(filelist, paths.concat(vocfolder, imglist:readString('*l')..'.jpg'))
end

function foo:generateSample(idx)
   require 'image'
   local path = filelist[idx]
   local target = 0
   local img0=image.load(path, 3, 'byte')
   
   local out=torch.ByteTensor(500,500,3):fill(0)
   local x = (#img0)[3]
   local y = (#img0)[2]
   local c = (#img0)[1]
   out:narrow(1,1,y):narrow(2,1,x):copy(img0:transpose(1,2):transpose(2,3))
   return out, torch.zeros(1):float()
end

foo:setBatchSize(batchsize) -- initial batch 4
foo:setSize(numimages)

foo:generateSet()


