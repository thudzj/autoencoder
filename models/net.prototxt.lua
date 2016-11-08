require 'ccn2'
local model = {}
table.insert(model, {'torch_transpose_dwhb', nn.Transpose({1,4},{1,3},{1,2})})
-- warning: module 'data [type 5]' not found
-- warning: module 'data_data_0_split [type 22]' not found
table.insert(model, {'torch_transpose_bdwh', nn.Transpose({4,1},{4,2},{4,3})})
table.insert(model, {'torch_view', nn.View(-1):setNumInputDims(3)})
table.insert(model, {'inner1', nn.Linear(784, 500)})
table.insert(model, {'inner1relu', nn.ReLU(true)})
table.insert(model, {'inner1drop', nn.Dropout(0.000000)})
table.insert(model, {'inner2', nn.Linear(500, 500)})
table.insert(model, {'inner2relu', nn.ReLU(true)})
table.insert(model, {'inner2drop', nn.Dropout(0.000000)})
table.insert(model, {'inner3', nn.Linear(500, 2000)})
table.insert(model, {'inner3relu', nn.ReLU(true)})
table.insert(model, {'inner3drop', nn.Dropout(0.000000)})
table.insert(model, {'output', nn.Linear(2000, 10)})
table.insert(model, {'outputdrop', nn.Dropout(0.000000)})
table.insert(model, {'d_inner3', nn.Linear(10, 2000)})
table.insert(model, {'d_inner3relu', nn.ReLU(true)})
table.insert(model, {'d_inner3drop', nn.Dropout(0.000000)})
table.insert(model, {'d_inner2', nn.Linear(2000, 500)})
table.insert(model, {'d_inner2relu', nn.ReLU(true)})
table.insert(model, {'d_inner2drop', nn.Dropout(0.000000)})
table.insert(model, {'d_inner1', nn.Linear(500, 500)})
table.insert(model, {'d_inner1relu', nn.ReLU(true)})
table.insert(model, {'d_inner1drop', nn.Dropout(0.000000)})
table.insert(model, {'d_data', nn.Linear(500, 784)})
-- warning: module 'pt_loss [type 7]' not found
return model