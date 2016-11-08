require 'cephes'

-- calculate the probability of a table
table_probability = function(h_i_det, h_i, h_j, d, kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
    -- print(h_i_mean, h_j_mean, h_ij_mean, nk_i_in, nk_j_in, nk_ij_in, lambda, d)
    local nk_i = h_i:size(1)
    local nk_j = h_j:size(1)
    local nk_ij = nk_i + nk_j
    local h_ij = torch.cat(h_i, h_j, 1)

    local result = math.pow((kappa_0+nk_i)*(kappa_0+nk_j)/kappa_0/(kappa_0+nk_ij), d/2.0)
    if result == math.huge or result ~= result then
        print('step1', nk_i, nk_j, nk_ij)
    end
    for i = 1, d do
        if cephes.beta((nu_0+nk_i+1-i)/2, (nu_0+nk_j+1-i)/2) == 0 then
            print(nk_i, nk_j)
            print('gg 0')
        end
        result = result * cephes.beta((nu_0+nk_ij+1-i)/2, (nu_0+1-i)/2) / cephes.beta((nu_0+nk_i+1-i)/2, (nu_0+nk_j+1-i)/2)
    end

    if result == math.huge or result ~= result then
        print('step2',nk_i, nk_j, nk_ij)
    end

    local h_mean = torch.mean(h_j, 1):view(1,d)
    local h_ = h_j - h_mean:expandAs(h_j)
    local Lambda_nk = Lambda_0 + (h_:transpose(1,2))*h_ + (h_mean:transpose(1,2) - mu_0:view(d, 1)) * (h_mean - mu_0:view(1, d)) 
                * nk_j * kappa_0 / (kappa_0 + nk_j)
    local Lambda_nk_e = torch.symeig(Lambda_nk:double(), 'N')
    local h_j_det = torch.prod(Lambda_nk_e)
    
    local h_mean = torch.mean(h_ij, 1):view(1,d)
    local h_ = h_ij - h_mean:expandAs(h_ij)
    local Lambda_nk = Lambda_0 + (h_:transpose(1,2))*h_ + (h_mean:transpose(1,2) - mu_0:view(d, 1)) * (h_mean - mu_0:view(1, d)) 
                * nk_ij * kappa_0 / (kappa_0 + nk_ij)
    local Lambda_nk_e = torch.symeig(Lambda_nk:double(), 'N')
    local h_ij_det = torch.prod(Lambda_nk_e)
    if h_ij_det == 0 then
        print(Lambda_nk)
        print('gg 1')
    end

    result = result * math.pow(h_i_det*h_j_det/h_ij_det, nu_0/2) * math.pow(h_i_det/h_ij_det, nk_i/2) * math.pow(h_j_det/h_ij_det, nk_j/2) / Lambda_0_det_pow_nu0_div_2
    if result == math.huge or result ~= result then
        print('step3', h_i_det, h_j_det, h_ij_det, nk_i, nk_j)
    end
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

gibbs_sample = function(z, belong, connect, tables, connected, n_tables, alpha, kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
    n_points = z:size(1)
    d = z:size(2)
    for i = 1, n_points do
        -- cache the proportion in the part 3 of equation 7
        local proportion = {}

        -- multinomial distribution
        local probability = torch.DoubleTensor(i)
        -- patition Z
        local sum_pro = 0.0

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
            -- clear the probability cache of table belong[i]
            n_tables = n_tables + 1
        end
        -- 3. calculate the table_probability of the new table
        local nk_i = #tables[i]
        local h_i = torch.CudaTensor(nk_i, d)
        for j = 1, nk_i do
            h_i[j] = z[tables[i][j]]
        end
        local h_mean = torch.mean(h_i, 1):view(1,d)
        local h_ = h_i - h_mean:expandAs(h_i)
        local Lambda_nk = Lambda_0 + (h_:transpose(1,2))*h_ + (h_mean:transpose(1,2) - mu_0:view(d, 1)) * (h_mean - mu_0:view(1, d)) 
                * nk_i * kappa_0 / (kappa_0 + nk_i)
        local Lambda_nk_e = torch.symeig(Lambda_nk:double(), 'N')
        local h_i_det = torch.prod(Lambda_nk_e)

        assert(belong[i] == i, "i must be the first of the new table")
        -- sample the new c_i
        for j = 1, i - 1 do
            assert(belong[i] ~= belong[j], "belong[i] ~= belong[j]")
            -- if not in cache
            if proportion[belong[j]] == nil then
                local nk_j = #tables[belong[j]]
                local h_j = torch.CudaTensor(nk_j, d)
                for jj = 1, nk_j do
                    h_j[jj] = z[tables[belong[j]][jj]]
                end
                
                proportion[belong[j]] = table_probability(h_i_det, h_i, h_j, d, kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
            end
            
            -- equation 7, part 3
            probability[j] = proportion[belong[j]] * math.exp(-torch.norm(z[i] - z[j])/alpha)
            if probability[j] == math.huge then
                print(proportion[belong[j]], -torch.norm(z[i] - z[j])/alpha)
                print("!!!Attention: table's probability is inf now!!!")
            end
                
            sum_pro = sum_pro + probability[j]
        end

        -- equation 7, part 1
        probability[i] = 1
        sum_pro = sum_pro + 1

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
        -- if target == i then
        --     print(probability)
        -- end

        -- assign the new connect
        -- print(i, target, sum_pro, probability[1])
        if target == 0 then
            print(i, probability)
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

cal_gradient = function(z, connect, tables, alpha, kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
    n_points = z:size(1)
    d = z:size(2)
    -- print(string.format("   [%s], calculating gradient of DDCRP", os.date("%c", os.time())))
    local dz = torch.CudaTensor(z:size()):fill(0)
    local sum_f_ii = torch.CudaTensor(n_points):fill(0)
    -- set sum_f_ii to be: sum^i_1(distanc_{i, j})
    -- print(string.format("[%s], start precompute", os.date("%c", os.time())))
    for i = 1, n_points do
        sum_f_ii[i] = torch.sum(torch.exp(-torch.norm(z[i]:view(1,d):expandAs(z[{{1, i}}]) - z[{{1, i}}], 2, 2)/alpha))
    end
    local l1_loss = 0
    -- part 1 of d(l_1)
    -- print(string.format("[%s], start part 1 of d(l_1)", os.date("%c", os.time())))
    for i = 1, n_points do
        l1_loss = l1_loss + torch.norm(z[i] - z[connect[i]])/alpha + math.log(sum_f_ii[i])
        if i ~= connect[i] then
            local tmp = (z[i] - z[connect[i]]) / torch.norm(z[i] - z[connect[i]])
            dz[i] = dz[i] + tmp/alpha
            dz[connect[i]] = dz[connect[i]] - tmp/alpha
        end
    end
    -- part 2 of d(l_1)
    -- print(string.format("[%s], start part 2 of d(l_1)", os.date("%c", os.time())))
    for i = 1, n_points do
        local tmp = z[i]:view(1,d):expandAs(z) - z
        local tmp_norm2 = torch.norm(tmp, 2, 2)
        if i > 1 then
            dz[i] = dz[i] - torch.sum(
                torch.cmul(
                    tmp[{{1, i-1}}],
                    torch.cdiv(
                        torch.exp(-tmp_norm2[{{1, i-1}}]/alpha), 
                        alpha*tmp_norm2[{{1, i-1}}]
                    ):expandAs(tmp[{{1, i-1}}])
                )
                /sum_f_ii[i], 
            1)
        end

        if i < n_points then
            dz[i] = dz[i] - torch.sum(
                torch.cmul(
                    tmp[{{i+1, n_points}}],
                    torch.cdiv(
                        torch.exp(-tmp_norm2[{{i+1, n_points}}]/alpha), 
                        alpha*torch.cmul(
                            tmp_norm2[{{i+1, n_points}}],
                            sum_f_ii[{{i+1, n_points}}]
                        )
                    ):expandAs(tmp[{{i+1, n_points}}])
                ), 
            1)
        end
    end
    -- d(l_2)
    local l2_loss = 0
    for table_id, table in pairs(tables) do
        local nk = #table
        if nk > 0 then
            local h = torch.CudaTensor(nk, d)
            for i = 1, nk do
                h[i] = z[table[i]]
            end
            local h_mean = torch.mean(h, 1):view(1,d)
            local h_ = h - h_mean:expandAs(h)
            local Lambda_nk = Lambda_0 + (h_:transpose(1,2))*h_ + 
                    (h_mean:transpose(1,2) - mu_0:view(d, 1)) * (h_mean - mu_0:view(1, d))*nk*kappa_0/(kappa_0+nk)
            
            local Lambda_nk_inv = torch.inverse(Lambda_nk)
            local dh = torch.CudaTensor(d, d)
            for i = 1, nk do
                for j = 1, d do
                    dh:fill(0)
                    -- combine equation 15 and 16
                    local tmp = h[i] - h_mean:view(d) + kappa_0 / (kappa_0 + nk) * (h_mean:view(d) - mu_0)
                    dh[j] = dh[j] + tmp
                    dh[{{}, j}] = dh[{{}, j}] + tmp
                    dz[table[i]][j] = dz[table[i]][j] + (nu_0 + nk) / 2 * torch.trace(Lambda_nk_inv * dh)
                end
            end

            local Lambda_nk_e = torch.symeig(Lambda_nk:double(), 'N')
            local h_det = torch.prod(Lambda_nk_e)
            l2_loss = l2_loss + nk*d*0.5*math.log(math.pi) + d*0.5*math.log((kappa_0+nk)/kappa_0) - math.log(Lambda_0_det_pow_nu0_div_2)
                + (nu_0+nk)*0.5*math.log(h_det)

            for i = 1, d do
                l2_loss = l2_loss + cephes.lgam((nu_0+1-i)/2) - cephes.lgam((nu_0+nk+1-j)/2)
            end
        end
    end

    return l1_loss, l2_loss, dz
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

    print (string.format('[%s], jointly training, epoch: %d, current loss: %4f, purity: %4f, MI: %4f, VI: %4f, RI: %4f', os.date("%c", os.time()), epoch, fs[1], purity, MI, (h_o+h_c)-2*MI, RI))
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