function [MSE1, EMSE1, w1] = diffusion_LMS_noncop(step_size, nodes, nodes_target, w1, M, N, N_ITER, metropolis, MSE1, EMSE1)

psi_b = zeros(N * M, 1);
local_w1 = zeros(size(w1));     

global_w1 = w1;                   

for f = 1:min(N_ITER, N_ITER - M)
    for node = 1:N
        pos_init = (node - 1) * M + 1;
        pos_end = node * M;
        psi = psi_b(pos_init:pos_end);
        data_vec = reshape(nodes(f:f+M-1, node), [], 1);
        
        if length(data_vec) < M
            data_vec = [data_vec; zeros(M - length(data_vec), 1)];
        end
        if sum(abs(data_vec)) == 0
            continue;
        end

        dki = nodes_target(f + 1);
        w_node = w1(node);  
        pki = w_node' * data_vec;  
    
        eki = dki - pki; 
        eaki = data_vec' * (w_node - psi);  

        %EMSE1(node, f) = (EMSE1(node, f) * (f - 1) + norm(eaki)^2) / f;
        MSE1(node, f) = (MSE1(node, f) * (f - 1) + norm(eki)^2) / f;

        step_size(node) = 0.004 / (1 + 0.0001 * f);  

        psi = psi + step_size(node) * eki .* data_vec;
        psi_b(pos_init:pos_end) = psi;

        for m = 1:M
            local_w1((node - 1) * M + m) = psi(m);  
        end
    end

    psi_matrix = reshape(psi_b, M, N);  
    psi_matrix = psi_matrix * metropolis;  
    psi_b = reshape(psi_matrix, N * M, 1);
    
    global_w1 = mean(local_w1, 2);  

    w1 = global_w1;
end

end