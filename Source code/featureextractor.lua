-- changes in weight file, featex comments, i=1:20 is same as number of featex lines

require 'cunxn'
featex=nxn.Sequential()


--foo=torch.load('/data/gpuuser1/weaksup_sk/netsaves/5_layer_15mb.t7')
foo=torch.load('/data/gpuuser1/weaksup_sk/netsaves/5_layer_15mb.t7')

--foo=torch.load('/data/gpuuser1/weaksup_sk/netsaves/6_layer_240mb.t7')


featex:add(nxn.SpatialConvolution(3,96,11,11,4,4,2,1,2,1))
featex:add(nxn.ReLU())
featex:add(nxn.CrossMapNormalization())
featex:add(nxn.SpatialMaxPooling(3,3,2,2))
featex:add(nxn.SpatialConvolution(96,256,5,5,1,1,2,2,2,2))
featex:add(nxn.ReLU())
featex:add(nxn.CrossMapNormalization())
featex:add(nxn.SpatialMaxPooling(3,3,2,2))
featex:add(nxn.SpatialConvolution(256,384,3,3,1,1,1,1,1,1))
featex:add(nxn.ReLU())
featex:add(nxn.SpatialConvolution(384,384,3,3,1,1,1,1,1,1))
featex:add(nxn.ReLU())
featex:add(nxn.SpatialConvolution(384,256,3,3,1,1,1,1,1,1))
featex:add(nxn.ReLU())
featex:add(nxn.SpatialMaxPooling(3,3,2,2))
--featex:add(nxn.SpatialConvolution(256,6144,6,6,1,1,0,0,0,0)) --256,6144
--featex:add(nxn.ReLU())
--featex:add(nxn.Affine(0.5))
--featex:add(nxn.SpatialConvolution(6144,256,1,1,1,1,0,0,0,0)) --6144
--featex:add(nxn.ReLU())


j=1
for i=1,15 do
   if featex.modules[i].weight then 
      featex.modules[i].weight=foo[j]:transpose(1,2):contiguous()
	print(i,j)
      j=j+1      
      featex.modules[i].bias=foo[j]
	print(i,j)
      j=j+1	
   end
end


