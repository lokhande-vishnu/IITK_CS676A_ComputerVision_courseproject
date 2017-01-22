-- Changes in numimages, batch size, images directory, target directory, clasmap, trainval, select index 
-- set image size as well


require 'cunxn'
require 'image'

ds=nxn.Dataset()

ds:setTargetDir('../datasets/countmap_500_50100')
imagesfolder='/data/gpuuser1/voc12-trainval/VOC2012/JPEGImages/'
numimages=500
batchsize=4

-- the goal here is to create the ds:generateSample(idx) function
-- this function should return the idx-th sample
-- all samples should be of fixed size (so we put them in batches)
-- the rest is done within the Dataset class


foo=torch.load('classmaps/countmap_500_50100.t7')
labelmap=foo.labels -- labelmap=foo.labels
xxx=torch.DiskFile('guides/vishnu.txt', 'r')

datatable={}

for idx=1,numimages do -- initial 11540
   table.insert(datatable, {paths.concat(imagesfolder, xxx:readString('*l')..'.jpg'), labelmap:select(2,idx):float()}) -- changed the select function
end

ds.datatable=datatable

function ds:generateSample(idx)
--   local foo=self.dataTable[idx]
   local foo=self.datatable[idx]
   local img0=image.load(foo[1], 3, 'byte')
   local out=torch.ByteTensor(50,100,3) --initial 500
   local x = (#img0)[3]
   local y = (#img0)[2]
   local c = (#img0)[1]
   out:narrow(1,1,y):narrow(2,1,x):copy(img0:transpose(1,2):transpose(2,3))
   return out, foo[2]:float() --:add(-0.5):mul(2)
end




-- then, some parameters
ds:setSize(numimages) --initial 11540
ds:setBatchSize(batchsize) -- initial batch 16
ds:shuffleOrder()

ds:generateSet()



-- resume generation : 
--if false then
--	ds=nxn.Dataset.loadSet('../datasets/voc12-full-weak-dataset')
--	ds:generateSet()
--end
