max_pooled_scores = torch.load('maxpooledScores.t7')
true_scores = torch.load('truescore.t7')

max_pooled_scores_modified_1 = torch.Tensor(2308, 20)
for i=1,2308 do
	for j=1,20 do
		max_pooled_scores_modified_1[i][j] = math.tanh(max_pooled_scores[i][j])
	end
end

temp = torch.Tensor(20,1)


err = torch.zeros(2308,1) -- error for each image
err_sum = torch.zeros(1,1)	-- total error
for i=1,2308 do
	a=max_pooled_scores_modified_1[{i, {}}] -- size 20 x 1
	b=true_scores[{i, {}}] -- size 20 x 1
	for j=1,20 do
		temp[j] = math.log(1+math.exp(-a[j]*b[j]))
	end
	for j=1,20 do
		err[i] = err[i] + temp[j]
	end
end
for i=1,2308 do
	err_sum = err_sum + err[i]
end

torch.save('error_for_each_image.t7', err)
torch.save('total_error.t7', err_sum)

--calculating minimum error

err_min = torch.zeros(2308,1) -- error for each image
err_sum_min = torch.zeros(1,1)	-- total error
for i=1,2308 do
	a=true_scores[{i, {}}] -- size 20 x 1
	b=true_scores[{i, {}}] -- size 20 x 1
	for j=1,20 do
		temp[j] = math.log(1+math.exp(-a[j]*b[j]))
	end
	for j=1,20 do
		err_min[i] = err_min[i] + temp[j]
	end
end
for i=1,2308 do
	err_sum_min = err_sum_min + err_min[i]
end

torch.save('min_error_for_each_image.t7', err_min)
torch.save('min_total_error.t7', err_sum_min)

max_pooled_scores_modified_2 = torch.Tensor(2308, 20)
for i=1,2308 do
	for j=1,20 do
		if max_pooled_scores[i][j] >= 0 then
			max_pooled_scores_modified_2[i][j] = 1
		else
			max_pooled_scores_modified_2[i][j] = -1
		end
	end
end

predictions = max_pooled_scores_modified_2
truth = true_scores
require('optim')

conf1 = optim.ConfusionMatrix( {'aeroplane','not_aeroplane'} )
conf1:zero() 
conf1:batchAdd(predictions[{{}, 1}], truth[{{}, 1}])
torch.save('aeroplane_confmat.t7', conf1)

conf2 = optim.ConfusionMatrix( {'bicycle','not_bicycle'} )
conf2:zero() 
conf2:batchAdd(predictions[{{}, 2}], truth[{{}, 2}])
torch.save('bicycle_confmat.t7', conf2)

conf3 = optim.ConfusionMatrix( {'bird','not_bird'} )
conf3:zero() 
conf3:batchAdd(predictions[{{}, 3}], truth[{{}, 3}])
torch.save('bird_confmat.t7', conf3)

conf4 = optim.ConfusionMatrix( {'boat','not_boat'} )
conf4:zero() 
conf4:batchAdd(predictions[{{}, 4}], truth[{{}, 4}])
torch.save('boat_confmat.t7', conf4)

conf5 = optim.ConfusionMatrix( {'bottle','not_bottle'} )
conf5:zero() 
conf5:batchAdd(predictions[{{}, 5}], truth[{{}, 5}])
torch.save('bottle_confmat.t7', conf5)

conf6 = optim.ConfusionMatrix( {'bus','not_bus'} )
conf6:zero() 
conf6:batchAdd(predictions[{{}, 6}], truth[{{}, 6}])
torch.save('bus_confmat.t7', conf6)

conf7 = optim.ConfusionMatrix( {'car','not_car'} )
conf7:zero() 
conf7:batchAdd(predictions[{{}, 7}], truth[{{}, 7}])
torch.save('car_confmat.t7', conf7)

conf8 = optim.ConfusionMatrix( {'cat','not_cat'} )
conf8:zero() 
conf8:batchAdd(predictions[{{}, 8}], truth[{{}, 8}])
torch.save('cat_confmat.t7', conf8)

conf9 = optim.ConfusionMatrix( {'chair','not_chair'} )
conf9:zero() 
conf9:batchAdd(predictions[{{}, 9}], truth[{{}, 9}])
torch.save('chair_confmat.t7', conf9)

conf10 = optim.ConfusionMatrix( {'cow','not_cow'} )
conf10:zero() 
conf10:batchAdd(predictions[{{}, 10}], truth[{{}, 10}])
torch.save('cow_confmat.t7', conf10)

conf11 = optim.ConfusionMatrix( {'dining_table','not_dining_table'} )
conf11:zero() 
conf11:batchAdd(predictions[{{}, 11}], truth[{{}, 11}])
torch.save('dining_table_confmat.t7', conf11)

conf12 = optim.ConfusionMatrix( {'dog','not_dog'} )
conf12:zero() 
conf12:batchAdd(predictions[{{}, 12}], truth[{{}, 12}])
torch.save('dog_confmat.t7', conf12)

conf13 = optim.ConfusionMatrix( {'horse','not_horse'} )
conf13:zero() 
conf13:batchAdd(predictions[{{}, 13}], truth[{{}, 13}])
torch.save('horse_confmat.t7', conf13)

conf14 = optim.ConfusionMatrix( {'motorbike','not_motorbike'} )
conf14:zero() 
conf14:batchAdd(predictions[{{}, 14}], truth[{{}, 14}])
torch.save('motorbike_confmat.t7', conf14)

conf15 = optim.ConfusionMatrix( {'person','not_person'} )
conf15:zero() 
conf15:batchAdd(predictions[{{}, 15}], truth[{{}, 15}])
torch.save('person_confmat.t7', conf15)

conf16 = optim.ConfusionMatrix( {'potted_plant','not_potted_plant'} )
conf16:zero() 
conf16:batchAdd(predictions[{{}, 16}], truth[{{}, 16}])
torch.save('potted_plant_confmat.t7', conf16)

conf17 = optim.ConfusionMatrix( {'sheep','not_sheep'} )
conf17:zero() 
conf17:batchAdd(predictions[{{}, 17}], truth[{{}, 17}])
torch.save('sheep_confmat.t7', conf17)

conf18 = optim.ConfusionMatrix( {'sofa','not_sofa'} )
conf18:zero() 
conf18:batchAdd(predictions[{{}, 18}], truth[{{}, 18}])
torch.save('sofa_confmat.t7', conf18)

conf19 = optim.ConfusionMatrix( {'train','not_train'} )
conf19:zero() 
conf19:batchAdd(predictions[{{}, 19}], truth[{{}, 19}])
torch.save('train_confmat.t7', conf19)

conf20 = optim.ConfusionMatrix( {'tv_monitor','not_tv_monitor'} )
conf20:zero() 
conf20:batchAdd(predictions[{{}, 20}], truth[{{}, 20}])
torch.save('tv_monitor_confmat.t7', conf20)

confmat = conf1.mat + conf2.mat + conf3.mat + conf4.mat + conf5.mat + conf6.mat + conf7.mat + conf8.mat + conf9.mat + conf10.mat + conf11.mat + conf12.mat + conf13.mat + conf14.mat + conf15.mat + conf16.mat + conf17.mat + conf18.mat + conf19.mat + conf20.mat

mat1 = conf1.mat
mat2 = conf2.mat
mat3 = conf3.mat
mat4 = conf4.mat
mat5 = conf5.mat
mat6 = conf6.mat
mat7 = conf7.mat
mat8 = conf8.mat
mat9 = conf9.mat
mat10 = conf10.mat
mat11 = conf11.mat
mat12 = conf12.mat
mat13 = conf13.mat
mat14 = conf14.mat
mat15 = conf15.mat
mat16 = conf16.mat
mat17 = conf17.mat
mat18 = conf18.mat
mat19 = conf19.mat
mat20 = conf20.mat

precision1 = mat1[1][1]/(mat1[1][1] + mat1[1][2])
precision2 = mat2[1][1]/(mat2[1][1] + mat2[1][2])
precision3 = mat3[1][1]/(mat3[1][1] + mat3[1][2])
precision4 = mat4[1][1]/(mat4[1][1] + mat4[1][2])
precision5 = mat5[1][1]/(mat5[1][1] + mat5[1][2])
precision6 = mat6[1][1]/(mat6[1][1] + mat6[1][2])
precision7 = mat7[1][1]/(mat7[1][1] + mat7[1][2])
precision8 = mat8[1][1]/(mat8[1][1] + mat8[1][2])
precision9 = mat9[1][1]/(mat9[1][1] + mat9[1][2])
precision10 = mat10[1][1]/(mat10[1][1] + mat10[1][2])
precision11 = mat11[1][1]/(mat11[1][1] + mat11[1][2])
precision12 = mat12[1][1]/(mat12[1][1] + mat12[1][2])
precision13 = mat13[1][1]/(mat13[1][1] + mat13[1][2])
precision14 = mat14[1][1]/(mat14[1][1] + mat14[1][2])
precision15 = mat15[1][1]/(mat15[1][1] + mat15[1][2])
precision16 = mat16[1][1]/(mat16[1][1] + mat16[1][2])
precision17 = mat17[1][1]/(mat17[1][1] + mat17[1][2])
precision18 = mat18[1][1]/(mat18[1][1] + mat18[1][2])
precision19 = mat19[1][1]/(mat19[1][1] + mat19[1][2])
precision20 = mat20[1][1]/(mat20[1][1] + mat20[1][2])

precision_sum = precision1 + precision2 + precision3 + precision4 + precision5 + precision6 + precision7 + precision8 + precision9 + precision10 + precision11 + precision12 + precision13 + precision14 + precision15 + precision16 + precision17 + precision18 + precision19 + precision20

torch.save('mAP.t7', (precision/20)*100)

precision = (precision_sum/20)*100