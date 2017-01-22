require 'cunxn'
require 'image'

ds=nxn.Dataset()

ds:setTargetDir('../datasets/train_voc07_5011_16')
imagesfolder='/data/gpuuser1/PASCALVOC2007/train/VOC2007/JPEGImages'

-- the goal here is to create the ds:generateSample(idx) function
-- this function should return the idx-th sample
-- all samples should be of fixed size (so we put them in batches)
-- the rest is done within the Dataset class


foo=torch.load('classmap_train_voc07.t7')
labelmap=foo.outmat
xxx=torch.DiskFile('trainval_voc07.txt', 'r')

datatable={}

for idx=1,5011 do
   table.insert(datatable, {paths.concat(imagesfolder, xxx:readString('*l')..'.jpg'), labelmap:select(2,idx):float()})
end

ds.datatable=datatable

function ds:generateSample(idx)
--   local foo=self.dataTable[idx]
   local foo=self.datatable[idx]
   local img0=image.load(foo[1], 3, 'byte')
   local out=torch.ByteTensor(500,500,3)
   local x = (#img0)[3]
   local y = (#img0)[2]
   local c = (#img0)[1]
   out:narrow(1,1,y):narrow(2,1,x):copy(img0:transpose(1,2):transpose(2,3))
   return out, foo[2]:float() --:add(-0.5):mul(2)
end




-- then, some parameters
ds:setSize(5011)
ds:setBatchSize(16)
ds:shuffleOrder()

ds:generateSet()



-- resume generation : 
if false then
	ds=nxn.Dataset.loadSet('../datasets/train_voc07_5011_16')
	ds:generateSet()
end
