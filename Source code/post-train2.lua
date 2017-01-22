require 'cunxn'


collectgarbage()

classes=20
torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('featureextractor.lua')

-- init model :
model = nxn.NeuralNet()

-- create adaptation layer
vocnet=nxn.Sequential()
vocnet:add(nxn.Dropmap(0.5))
transfer1=nxn.SpatialConvolution(6144,2048,3,3,1,1,1,1,1,1)
transfer1:setName('transfer1')
vocnet:add(transfer1)
vocnet:add(nxn.ReLU())
vocnet:add(nxn.Dropmap(0.5))
transfer2=nxn.SpatialConvolution(2048,classes,3,3,1,1,1,1,1,1)
transfer2:setName('transfer2')
vocnet:add(transfer2)
vocnet:add(nxn.SpatialGlobalMaxPooling())
vocnet:add(nxn.Reshape(classes))



-- create whole network
net=nxn.Sequential()

-- do jittering
jittering=nxn.TexFunRandResize(0.7,1.4,1)
jittering:setName('jittering')
net:add(jittering)

-- add feature extractor
net:add(featex)

-- add adaptation layer
net:add(vocnet)

-- put network in model
model:setNetwork(net)

-- choose cost
crit=nxn.MultiClassNLLCriterion():float()
model:setCriterion(crit)





-- choose dataset
dataset=nxn.Dataset.loadSet('../datasets/voc12-full-weak-dataset')

-- put dataset in model
model:setDataset(dataset)

-- training set and validation set
model:setTrainsetRange(1,dataset:getNumBatches()-51)
model:setTestsetRange(dataset:getNumBatches()-50,dataset:getNumBatches())


function model:getBatch(batchidx)
   local b,t= self.dataset:getBatch(batchidx)
   t=t:narrow(2,1,classes):add(-0.5):mul(2)
   return b,t
end


function model:showExample(batchidx, imgidx)
   local b,t=self:getBatch(batchidx)
   image.display(b:select(1,imgidx):transpose(2,3):transpose(2,1))
   print(t:select(1,imgidx))
end




-- set checkpoint path
model:setCheckpoint('../netsaves/', 'checkpoint.t7')
if false then 
   model=torch.load('../netsaves/checkpoint.t7')
   model:resume()
end


-- training parameters 
-- "setLearningRate" is propagated through all modules of a container
model.network:setLearningRate(0)

-- if a name is set, then model('name') returns module with the same name in model
model('transfer1'):autoLR()
model('transfer2'):autoLR()
model('transfer1'):setMomentum(0)
model('transfer2'):setMomentum(0)
model('transfer1'):setWeightDecay(5e-4)
model('transfer2'):setWeightDecay(5e-4)

-- some memory optimization (dropouts and ReLUs are applied directly on the input matrices)
model.network:setInPlace(1)
model.network:setSaveMem(true)

if not qt then
function model:showL1Filters()
end

function model:plotError()
end
end

model:train(20, 671, 671)

