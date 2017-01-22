require 'cunxn'

model=torch.load('../netsaves/checkpoint.t7')
model:resume()

