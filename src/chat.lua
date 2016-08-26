require 'hdf5'
local lexer = require 'pl.lexer'
local stringx = require 'pl.stringx'

require 'beam'
require 'dict'
require 'cunn'
------------
-- Options
------------

cmd = torch.CmdLine()

-- File location
cmd:option('-model',    'seq2seq_lstm.t7.', 	[[Path to model .t7 file]])
cmd:option('-lm',       'lm_lstm.t7',       	[[Path to language model .t7 file]])
cmd:option('-targ_dict','data/demo.targ.dict', 	[[Path to target vocabulary (*.targ.dict file)]])

-- Beam search options
cmd:option('-k',			50, 	[[Beam size]])
cmd:option('-max_sent_l', 	20, [[Maximum sentence length. If any sequences in srcfile are longer
                               		than this then it will error out]])
cmd:option('-simple', 		0, 	[[If = 1, output prediction is simply the first time the top of the beam
                         			ends with an end-of-sentence token. If = 0, the model considers all 
                         			hypotheses that have been generated so far that ends with end-of-sentence 
                         			token and takes the highest scoring of all of them.]])
cmd:option('-allow_unk', 	0, 	[[If = 1, prediction can include UNK tokens.]])
cmd:option('-antilm',		0, 	[[If = 1, prediction limits scoring contribution from earlier input.]])
cmd:option('-gamma',		3,	[[Number of initial word probabilities to discount from sequence probability.]])
cmd:option('-lambda',		0.45,[[Discount on initial word probabilities while using antiLM.]])
cmd:option('-len_reward',       2.5,[[Discount on initial word probabilities while using antiLM.]])
cmd:option('-k2',       40,[[Discount on initial word probabilities while using antiLM.]])

cmd:option('-decay',       0.9,[[Decay rate of lambda]])

opt = cmd:parse(arg)

------------
-- Misc
------------

function clean_input(sent)
    -- First split on spaces
    local split = stringx.split(string.lower(sent))
    local clean = {}

    -- Now partition on punctuation
    for i = 1, #split do
        for t, v in lexer.lua(split[i]) do
            table.insert(clean, v)
        end
    end

    return table.concat(clean, ' ')
end

function prep_input(sent)
    local clean = clean_input(sent)
    return sent2wordidx(clean, word2idx_targ)
end

function build_context(dialogue, length)
    local ctx = torch.LongTensor(1)
    local start = 1
    if #dialogue == 1 then return pad_both(dialogue[1]) end
    if #dialogue > length then start = #dialogue - length + 1 end
    
    -- Concatenate prior context, separated by utterance end tokens
    for i = start, #dialogue do
        local sep = torch.LongTensor({END_UTTERANCE})
        ctx = ctx:cat(dialogue[i]:cat(sep))
    end

    -- Remove inixial idx and final separator, then apply start and end tokens
    return pad_both(remove_pad(ctx))
end

------------
-- Chat
------------

function chat(sbeam)
    print("\nYou're now chatting with " .. opt.model .. ". Say hello!\n")

	local dialogue = {}
	local chatting = true
    local ctx_length = 2 -- For MovieTriples, use 2 prior utterances

    while chatting do
        -- Get next user input
        local response
        repeat
            -- io.write('you: ')
            io.flush()
            response = io.read()
        until response ~= nil and strip(response) ~= ''

        local prepped = prep_input(response)
        table.insert(dialogue, prepped)

        -- Generate contextual response
        local ctx = build_context(dialogue, ctx_length)

        -- Pick the best one
        -- local pred = remove_pad(sbeam:generate_map(ctx))

        -- Or pick randomly from k best?
        local k_best, scores = sbeam:generate_k(opt.k, ctx)
        local denom = 0
        for i = 1, #k_best do
            denom = denom + torch.exp(scores[i])
        end
        for i = 1, #k_best do
            scores[i] = torch.exp(scores[i]) / denom
        end
        local score_tensor = torch.DoubleTensor(scores)

        -- local pred = remove_pad(k_best[torch.multinomial(score_tensor, 1, true)[1]])
                local pred = remove_pad(k_best[math.random(#k_best)])

        -- local pred = remove_pad(k_best[1])

        local pred_sent = wordidx2sent(pred, idx2word_targ, false)
        table.insert(dialogue, pred)
        print('\n' .. pred_sent .. '\n')

        -- for i = 1, #k_best do
        --     pred_sent = wordidx2sent(remove_pad(k_best[i]), idx2word_targ, false)
        --     print('PRED (' .. scores[i] .. '): ' .. pred_sent)
        -- end
        print('')

        -- TODO: add logical way to end discourse
    end
end

------------
-- Set up
------------

function main()
    assert(path.exists(opt.model), 'model does not exist')

    -- Parse input params
    opt = cmd:parse(arg)

    print('Loading ' .. opt.model .. '...')
    local checkpoint = torch.load(opt.model)
    local lm
    if path.exists(opt.lm) then
        local lm_checkpoint = torch.load(opt.lm)
        lm = lm_checkpoint[1][2]
    end
    print('Done!')

    -- Load model and word2idx/idx2word dictionaries
    model, model_opt = checkpoint[1], checkpoint[2]
    idx2word_targ = idx2key(opt.targ_dict)
    word2idx_targ = flip_table(idx2word_targ)
    opt.layer_type = model_opt.layer_type

    -- Format model
    local enc = model[1]
    local dec = model[2]
    local enc_rnn = model[3]
    local dec_rnn = model[4]

    local m = {
        enc = enc,
        enc_rnn = enc_rnn,
        dec = dec,
        dec_rnn = dec_rnn
    }

    -- Initialize beam and start making chit-chat
    local sbeam = beam.new(opt, m, lm)
    chat(sbeam)
end

main()
