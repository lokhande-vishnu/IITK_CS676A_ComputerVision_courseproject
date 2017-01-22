-- Taking result for each scale and then taking average
-- changes in dataset directory, parameters, for loop if-else, classes

require 'cunxn'
if not model then 
   model=torch.load('../netsaves/checkpoint.t7')
   print('not model')
end
model:setTestMode(true)

scales = {0.5,0.7,1,1.4,2,2.8}
--scales = {1}

testset=nxn.Dataset.loadSet('../datasets/count_501_4')
model:setDataset(testset)

function model:getBatch(batchidx)
   local b,t= self.dataset:getBatch(batchidx)
   return b,t
end

numimages=500
batchsize=4 --4
numbatches=math.ceil(numimages/batchsize) -- 2748
last_batchsize=numimages%batchsize
if last_batchsize == 0 then last_batchsize = batchsize
end


outtable={}

classes=100*150
restable=torch.zeros(testset.numSamples,classes):float()
tgttable=torch.zeros(testset.numSamples,classes):float()
for _, scale in ipairs(scales) do
   print('scale : '..scale)
   model('jittering'):setScales(scale,scale,scale)
   
   restable:fill(0)

   for i=1,numbatches do
      print(i)
      b,tgt=model:getTestBatch(i)
      model:forward(b)
      out=model.network.output
      
      if i==numbatches then restable:narrow(1,(i-1)*batchsize+1,last_batchsize):add(out:float())
      else restable:narrow(1,(i-1)*batchsize+1,batchsize):add(out:float())
      end
      --tgttable:narrow(1,(i-1)*4+1,4):copy(tgt)
   end
   -- narrow(1,2,3) function picks a part of the table (2 to width 3) from dimension 1 (column)  
   -- The line then appends outs entries to that


   outtable[scale]=restable:clone() -- suplicates restable, otherwise behaves like a pointer
	-- we save the results for each scale, just in case
   torch.save('../results/outtable'..scale..'.t7', outtable[scale])
end

torch.save('../results/testsetmultiscale.t7', outtable)

-- in case it crashes, you should be able to retrieve results with this
savedtable={}
for _, scale in ipairs(scales) do
	if outtable[scale] then 
		savedtable[scale]=outtable[scale]
	else
		savedtable[scale]=torch.load('../results/outtable'..scale..'.t7')
	end
end

count=0
for a,b in pairs(savedtable) do count=count+1 end

resulttensor=torch.Tensor(count, numimages, classes)

count=1
for a,b in pairs(savedtable) do 
   resulttensor:select(1,count):copy(b) --[Tensor] select(dim, index) tensor slice
   count=count+1
end

maxpooledScores=resulttensor:mean(1):select(1,1)

torch.save('../results/maxpooledScores.t7', maxpooledScores)

-- maxpooledScores contains the 20 classes scores for the 10991 images of the test set in a 10991*20 matrix



