require 'itorch'
require 'cephes'
require 'distributions'

local function logp_studentT(x, mu, chol_sigma, v)
   local lgam = cephes.lgam
   local d = mu:size(1)

   local inv_sigma = torch.potri(chol_sigma)
   local log_det_sigma = torch.diag(chol_sigma):log():sum() * 2
   -- local inv_sigma = torch.inverse(chol_sigma)
   -- local log_det_sigma = distributions.util.logdet(chol_sigma)

   local s = lgam((v+d)/2) - lgam(v/2)
    if s ~= s then
     assert(false, 'gg___1')
   end
   s = s -
      (log_det_sigma / 2) -
      (torch.log(v) * d/2) -
      (torch.log(math.pi) * d/2)

    if s ~= s then
        print(torch.diag(chol_sigma):log())
        print((log_det_sigma / 2))
        print(torch.log(v) * d/2)
     assert(false, 'gg_0')
   end

   local x_c = x - mu
   local result = torch.log(1+(x_c * (inv_sigma * x_c) * (1/v))) * (v+d)/2
   result = s - result

   if result ~= result then
    assert(false, 'gg_1')
   end

   if result ~= result then
     assert(false, 'gg')
   end
   return result
end


-- calculate the minus log likelihood of a table
table_probability = function(h, mu_0, kappa_0, lambda_0, nu_0, det_lambda_0)
    local n = h:size(1)
    local d = h:size(2)
    local h_mean = torch.mean(h, 1):view(1, d)
    local h_ = h - h_mean:expandAs(h)
    local lambda_n = lambda_0 + (h_:t())*h_ + n*kappa_0/(kappa_0 + n)*(h_mean:t() - mu_0)*(h_mean - mu_0:t()) 

    local result = n*d/2*math.log(math.pi)+d/2*math.log((kappa_0+n)/kappa_0)-nu_0/2*det_lambda_0
    result = result + (nu_0+n)/2*distributions.util.logdet(lambda_n:double())
    result = result + cephes.lmvgam(nu_0/2,d) - cephes.lmvgam((nu_0+n)/2,d)
    
    -- assert(result >= 0, 'line 16')
    return result
end

-- detach a connect and then generate a new table
split_table = function(tables, belong, connected, table_id, new_table_id, point_id)
    table.insert(tables[new_table_id], point_id)
    belong[point_id] = new_table_id
    for j = 1, #tables[table_id] do
        if tables[table_id][j] == point_id then
            table.remove(tables[table_id], j)
            break
        end
    end
    for i = 1, #connected[point_id] do
        if connected[point_id][i] ~= point_id then
            split_table(tables, belong, connected, table_id, new_table_id, connected[point_id][i])
        end
    end
end

gibbs_sample = function(z, belong, connect, tables, connected, n_tables, alpha, mu_0, kappa_0, lambda_0, nu_0, det_lambda_0)
    local n_points = z:size(1)
    local d = z:size(2)
    local mu_0_d = mu_0:double()
    local lambda_0_d = lambda_0:double()
    for i = 1, n_points do
        -- cache the proportion in the part 3 of equation 7
        local log_p_cache = {}
        -- multinomial distribution
        local probability = torch.DoubleTensor(i)

        -- detach the table
        -- 1. remove the connect of connected[connect[i]]
        for j = 1, #connected[connect[i]] do
            if connected[connect[i]][j] == i then
                table.remove(connected[connect[i]], j)
                break
            end
        end
        -- 2. if c_i is not self-connected, generative a new table
        if connect[i] ~= i then
            tables[i] = {}
            split_table(tables, belong, connected, belong[i], i, i)
            n_tables = n_tables + 1
        end
        -- 3. calculate the table_probability of the new table
        local nk_i = #tables[i]
        local h_i = torch.DoubleTensor(nk_i, d)
        for j = 1, nk_i do
            h_i[j] = z[tables[i][j]]
        end

        assert(belong[i] == i, "i must be the first of the new table")
        -- sample the new c_i
        for j = 1, i - 1 do
            assert(belong[i] ~= belong[j], "belong[i] ~= belong[j]")
            -- if not in cache
            if torch.norm(z[i] - z[j]) < alpha then
                if log_p_cache[belong[j]] == nil then
                    local nk_j = #tables[belong[j]]
                    local h_j = torch.DoubleTensor(nk_j, d)
                    for jj = 1, nk_j do
                        h_j[jj] = z[tables[belong[j]][jj]]
                    end
                    
                    log_p_cache[belong[j]] = table_probability(h_j, mu_0_d, kappa_0, lambda_0_d, nu_0, det_lambda_0)
                                           - table_probability(torch.cat(h_i, h_j, 1), mu_0_d, kappa_0, lambda_0_d, nu_0, det_lambda_0)
                end
                probability[j] = log_p_cache[belong[j]]
            else
                probability[j] = -math.huge
            end
        end

        -- equation 7, part 1
        probability[i] = 0
        probability = torch.exp(probability - torch.max(probability))
        local sum_pro = torch.sum(probability)

        -- gibbs sampling
        local U = torch.uniform(0, sum_pro)
        local u = 0.0
        local target = 0
        for j = 1, i do
            u = u + probability[j]
            if u > U then
                target = j
                break
            end
        end

        assert(target > 0, "Target below zero is invalid.")
        connect[i] = target
        table.insert(connected[target], i)

        -- connect the new c_i and then update the tables and table_probabulitys
        if i ~= belong[target] then
            assert(belong[target] < belong[i], "The new c_i must be connected to a former point")
            for j = 1, #tables[i] do
                table.insert(tables[belong[target]], tables[i][j])
                belong[tables[i][j]] = belong[target]
            end
            n_tables = n_tables - 1
            tables[i] = {}
        end
    end

    print(string.format("   [%s], gibbs sampling, # of points: %d, # of tables: %d", os.date("%c", os.time()), n_points, n_tables))
    return n_tables
end

crp_init = function(z, means, covs, tables, mu_0, kappa_0, lambda_0, nu_0)
    local d = z:size(2)
    for i, table in pairs(tables) do
        local n = #table
        local z_i = torch.CudaTensor(n, d)
        for j = 1, n do
            z_i[j] = z[table[j]]
        end
        means[i] = ((kappa_0 * mu_0:view(d) + n * torch.mean(z_i, 1)) / (kappa_0 + n))
        covs[i] = (lambda_0 + z_i:t() * z_i - (kappa_0+n) * means[i]:view(d,1) * means[i]:view(1,d) + kappa_0 * mu_0 * mu_0:t()):double()
        means[i] = means[i]:double()
    end
end

crp_gibbs_sample = function(z, means, covs, tables, belong, alpha, mu_0, kappa_0, lambda_0, nu_0)
    local n_points = z:size(1)
    local d = z:size(2)

    for i = 1, n_points do
        -- remove c_i
        if #tables[belong[i]] == 1 then
            table.remove(tables, belong[i])
            for j = 1, n_points do
                if belong[j] > belong[i] then
                    belong[j] = belong[j] - 1
                end
            end
        else
            local n = #tables[belong[i]]
            for j = 1, n do
                if tables[belong[i]][j] == i then
                    table.remove(tables[belong[i]], j)
                    break
                end
            end
            covs[belong[i]] = covs[belong[i]] + (kappa_0+n) * means[belong[i]]:view(d, 1) * means[belong[i]]:view(1, d) - z[i]:view(d, 1) * z[i]:view(1, d)
            means[belong[i]] = (means[belong[i]] * (kappa_0 + n) - z[i]) / (kappa_0 + n - 1)
            covs[belong[i]] = covs[belong[i]] - (kappa_0+n-1) * means[belong[i]]:view(d, 1) * means[belong[i]]:view(1, d)
        end

        -- calculate probability
        local probability = torch.DoubleTensor(#tables + 1):fill(0)
        for j, table in pairs(tables) do
            local n = #table
            assert(n > 0, 'line 181')
            probability[j] = n / (n_points - 1 + alpha) * math.exp(logp_studentT(z[i], means[j], covs[j]*(kappa_0+n+1)/(kappa_0+n)/(nu_0+n-d+1), nu_0+n-d+1))
        end
        probability[#tables + 1] = alpha / (n_points - 1 + alpha) * math.exp(logp_studentT(z[i], mu_0:view(d), lambda_0*(kappa_0+1)/kappa_0/(nu_0-d+1), nu_0-d+1))
        local sum_pro = torch.sum(probability)

        -- gibbs sampling
        local U = torch.uniform(0, sum_pro)
        local u = 0.0
        local target = 0
        for j = 1, #tables + 1 do
            u = u + probability[j]
            if u > U then
                target = j
                break
            end
        end

        -- create c_i
        if target < 1 or target > #tables + 1 then
            print(probability)
        end
        assert(target > 0 and target <= #tables + 1, 'line @213')
        belong[i] = target
        if target == #tables + 1 then
            tables[#tables + 1] = {i}
            means[#tables] = (kappa_0*mu_0 + z[i])/(kappa_0+1)
            covs[#tables] = lambda_0 + z[i]:view(d,1)*z[i]:view(1,d) - (kappa_0+1)*means[#tables]:view(d,1)*means[#tables]:view(1,d) + kappa_0*mu_0*mu_0:t()
        else
            local n = #tables[target]
            assert(n > 0, 'line 222')
            table.insert(tables[target], i)
            covs[target] = covs[target] + (kappa_0+n) * means[target]:view(d, 1) * means[target]:view(1, d) + z[i]:view(d, 1) * z[i]:view(1, d)
            means[target] = (means[target] * (kappa_0 + n) + z[i]) / (kappa_0 + n + 1)
            covs[target] = covs[target] - (kappa_0+n+1) * means[target]:view(d, 1) * means[target]:view(1, d)
        end
    end
    -- count tables
    print(string.format("   [%s], gibbs sampling, # of tables: %d", os.date("%c", os.time()), #tables))
    return (#tables)
end

cal_gradient = function(z, connect, tables, alpha, mu_0, kappa_0, lambda_0, nu_0, det_lambda_0)
    n_points = z:size(1)
    d = z:size(2)
    -- print(string.format("   [%s], calculating gradient of DDCRP", os.date("%c", os.time())))
    local dz = torch.CudaTensor(z:size()):fill(0)
    -- local sum_f_ii = torch.CudaTensor(n_points):fill(0)
    -- -- set sum_f_ii to be: sum^i_1(distanc_{i, j})
    -- -- print(string.format("[%s], start precompute", os.date("%c", os.time())))
    -- for i = 1, n_points do
    --     sum_f_ii[i] = torch.sum(torch.exp(-torch.norm(z[i]:view(1,d):expandAs(z[{{1, i}}]) - z[{{1, i}}], 2, 2)/alpha))
    -- end
    local l1_loss = 0
    -- part 1 of d(l_1)
    -- print(string.format("[%s], start part 1 of d(l_1)", os.date("%c", os.time())))
    -- for i = 1, n_points do
    --     l1_loss = l1_loss + torch.norm(z[i] - z[connect[i]])/alpha + math.log(sum_f_ii[i])
    --     if i ~= connect[i] then
    --         local tmp = (z[i] - z[connect[i]]) / torch.norm(z[i] - z[connect[i]])
    --         dz[i] = dz[i] + tmp/alpha
    --         dz[connect[i]] = dz[connect[i]] - tmp/alpha
    --     end
    -- end
    -- -- -- part 2 of d(l_1)
    -- -- -- print(string.format("[%s], start part 2 of d(l_1)", os.date("%c", os.time())))
    -- local tmpz = torch.CudaTensor(d)
    -- local dz_norm2 = torch.CudaTensor(n_points)
    -- for i = 1, n_points do
    --     local tmp = z[i]:view(1,d):expandAs(z) - z
    --     local tmp_norm2 = torch.norm(tmp, 2, 2)
    --     tmpz:fill(0)
    --     if i > 1 then
    --         tmpz = tmpz + torch.sum(
    --             torch.cmul(
    --                 tmp[{{1, i-1}}],
    --                 torch.cdiv(
    --                     torch.exp(-tmp_norm2[{{1, i-1}}]/alpha), 
    --                     alpha*tmp_norm2[{{1, i-1}}]
    --                 ):expandAs(tmp[{{1, i-1}}])
    --             )
    --             /sum_f_ii[i], 
    --         1)
    --     end

    --     if i < n_points then
    --         tmpz = tmpz + torch.sum(
    --             torch.cmul(
    --                 tmp[{{i+1, n_points}}],
    --                 torch.cdiv(
    --                     torch.exp(-tmp_norm2[{{i+1, n_points}}]/alpha), 
    --                     alpha*torch.cmul(
    --                         tmp_norm2[{{i+1, n_points}}],
    --                         sum_f_ii[{{i+1, n_points}}]
    --                     )
    --                 ):expandAs(tmp[{{i+1, n_points}}])
    --             ), 
    --         1)
    --     end

    --     dz[i] = dz[i] - tmpz
    --     dz_norm2[i] = torch.norm(tmpz)
    --     if dz_norm2[i] ~= dz_norm2[i] then
    --         print(i, tmpz)
    --     end
    -- end
    -- if ite == 3 then
    --     itorch.Plot():line(torch.range(1, n_points), dz_norm2:double(),'red','example'):legend(true):title(string.format("%d", epoch)):save(string.format('visualization/gradient_%d.html', epoch))
    -- end
    -- d(l_2)
    local l2_loss = 0
    local z_mean = torch.mean(z, 1):view(1,d)
    local S_b = 0
    local S_w = 0
    local table_mean = {}

    for table_id, table in pairs(tables) do
        local nk = #table
        if nk > 0 then
            local h = torch.CudaTensor(nk, d)
            for i = 1, nk do
                h[i] = z[table[i]]
            end
            local h_mean = torch.mean(h, 1):view(1,d)
            table_mean[table_id] = h_mean
            S_b = S_b + nk*torch.norm(h_mean - z_mean)^2
            local h_ = h - h_mean:expandAs(h)
            local lambda_nk = lambda_0 + (h_:t())*h_ + 
                    (h_mean:t() - mu_0) * (h_mean - mu_0:t())*nk*kappa_0/(kappa_0+nk)
            
            local lambda_nk_inv = torch.inverse(lambda_nk)
            local dh = torch.CudaTensor(d, d)
            for i = 1, nk do
                S_w = S_w + torch.norm(h_mean - h[i])^2
                for j = 1, d do
                    dh:fill(0)
                    -- combine equation 15 and 16
                    local tmp = h[i] - h_mean:view(d) + kappa_0/(kappa_0 + nk)*((h_mean - mu_0:t()):view(d))
                    dh[j] = dh[j] + tmp
                    dh[{{}, j}] = dh[{{}, j}] + tmp
                    dz[table[i]][j] = dz[table[i]][j] + (nu_0 + nk) / 2 * torch.trace(lambda_nk_inv * dh)
                end
            end

            l2_loss = l2_loss + table_probability(h, mu_0, kappa_0, lambda_0, nu_0, det_lambda_0)
        end
    end

    -- for table_id, table in pairs(tables) do
    --     local nk = #table
    --     if nk > 0 then
    --         local h = torch.CudaTensor(nk, d)
    --         for i = 1, nk do
    --             h[i] = z[table[i]]
    --         end
    --         local h_mean = torch.mean(h, 1):view(1,d)
            
    --         local h_ = (h - h_mean:expandAs(h))*lambda_inv + ((h_mean-mu_0:t())*torch.inverse(lambda_0*nk+lambda)):expandAs(h)
    --         -- (h-(nk*h_mean+k_0*mu_0:t()):expandAs(h)/(nk+k_0))*lambda

    --         l2_loss = l2_loss + table_probability(h, mu_0, lambda_0, lambda, det_lambda, lambda_inv, evalues:cuda(), 1) 
    --         --l2_loss - (nk/lambda)*(nk/lambda)/2/(nk/lambda+1)*(torch.norm(h_mean)^2) + (torch.norm(h)^2)/2/lambda + 0.5*d*math.log(nk/lambda+1) + 0.5*nk*d*math.log(2*math.pi*lambda)
            
    --         for i = 1, nk do
    --             dz[table[i]] = dz[table[i]] + h_[i]
    --         end
    --     end
    -- end

    -- if ite == 3 then
    --     itorch.Plot():line(torch.range(1, n_points), torch.norm(dz, 2, 2):view(n_points):double(),'red','l2_norm(dz)'):legend(true):title(string.format("%d", epoch)):save(string.format('visualization/dz_%d.html', epoch))
    -- end

    local S_loss = S_w / S_b
    local S_dz = torch.CudaTensor(z:size()):fill(0)
    for table_id, table in pairs(tables) do
        local nk = #table
        if nk > 0 then
            for i = 1, nk do
                S_dz[table[i]] = S_dz[table[i]] + (z[table[i]] - table_mean[table_id]) * 2 / S_b
                    - (table_mean[table_id] - z_mean) * 2 * S_w / S_b
            end
        end
    end

    return l1_loss, l2_loss, S_loss, dz, S_dz
end

evaluate_cluster = function(tables, label, epoch, fs, n_points, clusters)

    local h_c = 0
    local cls_10 = torch.DoubleTensor(clusters):fill(0)
    for cls = 1, clusters do
        local cnt = 0
        for ite = 1, n_points do
            if label[ite] == cls - 1 then
                cnt = cnt + 1
            end
        end
        cls_10[cls] = cnt
        h_c = h_c - cnt/n_points*math.log(cnt/n_points)/math.log(2)
        -- print(cnt/n_points, h_c)
    end

    local h_o = 0
    local MI = 0
    local tp_fp = 0
    local tp = 0
    local fn_tn = 0
    local purity = 0
    local fn = 0
    local clu_num = {}
    local clu_cnt_10 = {}
    for table_id, table_i in pairs(tables) do
        if #table_i > 0 then
            h_o = h_o - (#table_i)/n_points*math.log((#table_i)/n_points)/math.log(2)

            tp_fp = tp_fp + #table_i*(#table_i-1)/2
            for ite = 1, #clu_num do
                fn_tn = fn_tn + clu_num[ite] * (#table_i)
            end
            table.insert(clu_num, #table_i)

            local cnt_10 = torch.DoubleTensor(clusters):fill(0)
            for porint_id, point in pairs(table_i) do
                cnt_10[label[point]+1] = cnt_10[label[point]+1] + 1
            end
            purity = purity + torch.max(cnt_10) / n_points
            for cls = 1, clusters do
                if cnt_10[cls] > 0 then
                    MI = MI + cnt_10[cls]/n_points*math.log(n_points*cnt_10[cls]/(#table_i)/cls_10[cls])/math.log(2)
                end
                tp = tp + cnt_10[cls]*(cnt_10[cls]-1)/2

                for ite = 1, #clu_cnt_10 do
                    fn = fn + clu_cnt_10[ite][cls] * cnt_10[cls]
                end
            end
            table.insert(clu_cnt_10, cnt_10)
        end
    end

    local tn = fn_tn - fn
    local RI = (tp+tn) / (tp_fp+fn_tn)

    print (string.format('[%s], jointly training, epoch: %d, current loss: %4f, purity: %4f, MI: %4f,NMI: %4f, VI: %4f, RI: %4f', os.date("%c", os.time()), epoch, fs[1], purity, MI, 2*MI/(h_o+h_c),(h_o+h_c)-2*MI, RI))
end

visualize = function(labels, z, epoch)
    local x = torch.Tensor(labels:size())
    local y = torch.Tensor(labels:size())

    local cnt = 0
    local table_heads = nil
    local maps = {}
    for table_id, table in pairs(tables) do
        if #table > 0 then
            cnt = cnt + 1
            maps[cnt] = table_id
            if table_heads == nil then
                table_heads = z[table_id]:view(1, d)
            else
                table_heads = torch.cat(table_heads, z[table_id]:view(1,d), 1)
            end
        end
    end

    local mean = torch.mean(table_heads, 1) -- 1 x n
    local m = table_heads:size(1)
    local Xm = table_heads - torch.ones(m, 1) * mean
    Xm:div(math.sqrt(m - 1))
    vecs,s_,_ = torch.svd(Xm:t())
    X_hat = (table_heads - torch.ones(m,1) * mean) * vecs[{ {},{1, 2} }]

    for i = 1, #maps do
        x[maps[i]] = X_hat[i][1]
        y[maps[i]] = X_hat[i][2]
    end

    for table_id, table in pairs(tables) do
        for i = 1, #table do
            if table[i] ~= table_id then
                x[table[i]] = torch.randn(1)*0.1 + x[table_id]
                y[table[i]] = torch.randn(1)*0.1 + y[table_id]
            end
        end
    end

    itorch.Plot():gscatter(x,y,labels):title(string.format('Epoch %d clustering result(clusters: %d)', epoch, n_tables)):save(string.format('visualization/visualization_%d.html', epoch))
end