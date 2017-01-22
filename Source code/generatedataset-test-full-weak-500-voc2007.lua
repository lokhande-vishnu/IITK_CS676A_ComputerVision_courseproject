require 'cunxn'
foo=nxn.Dataset()



imglist=torch.DiskFile('test_voc07.txt', 'r')

vocfolder='/data/gpuuser1/PASCALVOC2007/test/VOC2007/JPEGImages'
foo:setTargetDir('../datasets/test_voc07_4952_4')

filelist={}
for i=1,4952 do
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

foo:setBatchSize(4)
foo:setSize(4952)

foo:generateSet()


