------------------------------------------------------------------------
-- train.lua
--
-- General training file for this project. 
--
-- The file contains all of the command line arguments/parameters that 
--      are used during training.
--     
-- This file loads in the command line arguments, loads in the functions 
--      that it needs, and then will either:
--          a) call the parent() function if the -parallel flag is on, 
--          b) call the main() function and run in serial.
------------------------------------------------------------------------

require 'data'

------------
-- Options
------------

cmd = torch.CmdLine()

-- Data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-data_file',    'data/conv-train.hdf5',     'Path to the training *.hdf5 file from preprocess.py')
cmd:option('-val_data_file','data/conv-val.hdf5',       'Path to validation *.hdf5 file from preprocess.py')
cmd:option('-save_file',    '',                         'Save file name (model will be saved as savefile_epochX_PPL.t7  where X is the X-th epoch and PPL is the validation perplexity')
cmd:option('-train_from',   '',                         'If training from a checkpoint then this is the path to the pretrained model.')
cmd:option('-load_red',   false,                        'If training a HRED model and loading parameters from subtle red model.')

-- RNN model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers',       2,      'Number of layers in the LSTM encoder/decoder')
cmd:option('-hidden_size',      300,    'Size of LSTM hidden states')
cmd:option('-word_vec_size',    300,    'Word embedding sizes')
cmd:option('-layer_type',       'lstm', 'Recurrent layer type (rnn, gru, lstm, bi)')
cmd:option('-model_type',       'red', 	'Model structure (red, hred)')
cmd:option('-utter_context',	2,		'Number of utterances in context')

-- cmd:option('-reverse_src',   0,      'If 1, reverse the source sequence. The original 
--                                      sequence-to-sequence paper found that this was crucial to 
--                                      achieving good performance, but with attention models this
--                                      does not seem necessary. Recommend leaving it to 0')
-- cmd:option('-init_dec',      1,      'Initialize the hidden/cell state of the decoder at time 
--                                      0 to be the last hidden/cell state of the encoder. If 0, 
--                                      the initial states of the decoder are set to zero vectors')

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

-- Optimization
cmd:option('-num_epochs',       10,     'Number of training epochs')
cmd:option('-start_epoch',      1,      'If loading from a checkpoint, the epoch from which to start')
cmd:option('-param_init',       0.1,    'Parameters are initialized over uniform distribution with support (-param_init, param_init)')
cmd:option('-learning_rate',    .01,    'Initial learning rate')
cmd:option('-ada_grad',         true,   'When true, update parameters using adagrad algorithm')
cmd:option('-max_grad_norm',    5,      'If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm')
cmd:option('-dropout',          0.3,    'Dropout probability. Dropout is applied between vertical LSTM stacks.')
cmd:option('-lr_decay',         0.5,    'Decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past the start_decay_at_limit')
cmd:option('-start_decay_at',   9,      'Start decay after this epoch')
cmd:option('-fix_word_vecs',    0,      'If = 1, fix lookup table word embeddings')
cmd:option('-beam_k',           5,      'K value to use with beam search')
cmd:option('-max_bleu',         4,      'The number of n-grams used in calculating the bleu score')

cmd:option('-pre_word_vecs',    '', 'If a valid path is specified, then this will load pretrained word embeddings (hdf5 file) on the encoder side. See README for specific formatting instructions.')

-- cmd:option('-curriculum',    0,      'For this many epochs, order the minibatches based on source
--                                       sequence length. Sometimes setting this to 1 will increase convergence speed.')

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid',    -1, 'Which gpu to use. -1 = use CPU')
cmd:option('-gpuid2',   -1, 'If this is >= 0, then the model will use two GPUs whereby the encoder is on the first GPU and the decoder is on the second GPU. This will allow you to train with bigger batches/models.')

-- Bookkeeping
cmd:option('-save_every',   1,      'Save every this many epochs')
cmd:option('-print_every',  5,      'Print stats after this many batches')
cmd:option('-seed',         3435,   'Seed for random initialization')

-- Parallel
cmd:option('-parallel',         false,  'When true, uses the parallel library to farm out sgd')
cmd:option('-n_proc',           2,      'The number of processes to farm out')
cmd:option('-setup_servers',    false,   'When true, executes code to setup external servers ')
cmd:option('-localhost',        false,   'When true, the farmed out processes will run through localhost')
cmd:option('-remote',           false,   'When true, the farmed out processes are run on remote servers. overrides localhost')
cmd:option('-torch_path',       '/Users/michaelfarrell/torch/install/bin/th',   'The path to the torch directory')
cmd:option('-extension',       '',   'The location from the home directory to the helper functions')
cmd:option('-username',       'michaelfarrell',   'The username for connecting to remote clients')
cmd:option('-add_to_path' ,     '/home/michaelfarrell/Distributed-SGD/lua-lua/End-To-End-Generative-Dialogue/src/?.lua;',  'A string that will be appended on to the front of the path')
cmd:option('-wait',         200,  'Waits to parallelize until below this threshold')
cmd:option('-batch_first_dimension', true,'If the input has been formatted to use the first dimension as batch, set this to true')
-- Used to update the path variable
require 'package'

-- Parse arguments
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)


if opt.save_file:len() == 0 then
    opt.model_id = opt.layer_type .. '_' .. opt.model_type
    if opt.data_file ~= 'data/conv-train.hdf5' then 
        opt.model_id = opt.model_id .. '_subtle'
    end
    if opt.fix_word_vecs then
        opt.model_id = opt.model_id  .. '_fix'
    end 
    if opt.hidden_size ~= 300 then 
        opt.model_id = opt.model_id .. '_hs' .. opt.hidden_size
    end
    if opt.word_vec_size ~= 300 then
        opt.model_id = opt.model_id .. '_wv' .. opt.word_vecs
    end
    if opt.ada_grad then 
        opt.model_id = opt.model_id .. '_ada'
    end
end
print(opt.model_id)

-- Add on location to path of new class if not already in path
package.path = opt.add_to_path .. package.path

-- The parent process function
function parent()
    -- Load in the class that runs the server
    server = require('sgd_server')

    -- Print from parent process
    parallel.print('Im the parent, my ID is: ',  parallel.id, ' and my IP: ', parallel.ip)

    -- Initialize Server from server.lua class
    param_server = server.new(opt)

    -- Run the server
    param_server:run()   
end

if opt.gpuid >= 0 then
    print('Using CUDA on GPU ' .. opt.gpuid .. '...')
    if opt.gpuid2 >= 0 then
        print('Using CUDA on second GPU ' .. opt.gpuid2 .. '...')
    end
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
end

-- Run in parallel
if opt.parallel then
    require 'parallel'

    -- Load in functions used for parallel
    opt.print = parallel.print
    -- opt.learning_rate = opt.learning_rate / opt.n_proc

    -- Protected execution of parllalel script:
    ok, err = pcall(parent)
    if not ok then print(err) parallel.close() end
else

    funcs = loadfile("model_functions.lua")
    funcs()
    
    opt.print = print
    
    -- Create the data loader classes
    local train_data, valid_data, opt = load_data(opt)
    
    -- Build
    local model, criterion = build()

    -- Train
    train(model, criterion, train_data, valid_data)

end