function [MSE, EMSE, w] = diffusion_LMS(step_size, nodes, nodes_target, w, M, N, N_ITER, metropolis, MSE, EMSE)

psi_b = zeros(N * M, 1);  
last_input_all_nodes = zeros(M * N, 1);  

WH_NL1 = @(x) tanh(0.1 * x);                  
WH_NL2 = @(y) sign(y) .* abs(y).^0.8;         

FIR_coeff = [0.55, -0.18, 0.09, -0.045, 0.02];  % Slightly reduced coefficients

FIR_len = length(FIR_coeff); 

global_w = w;                  
local_w = zeros(size(w));     

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
        data_vec_nl = WH_NL1(data_vec);

        % Apply FIR System ONLY for the Current Node 
        fir_output = conv(data_vec_nl, FIR_coeff, 'same');

        % Apply Post-FIR Nonlinear Transformation (WH_NL2)
        fir_output_nl = WH_NL2(fir_output);
        fir_output_nl = fir_output_nl(1:M);  

        dki = nodes_target(f + 1);
        w_node = w(node);  
        pki = w_node' * fir_output_nl;  % Apply LMS with Wiener-Hammerstein Transformed Data
    
        eki = dki - pki; 
        eaki = fir_output_nl' * (w_node - psi);  

        % Update Error Metrics
        %EMSE(node, f) = (EMSE(node, f) * (f - 1) + norm(eaki)^2) / f;
        MSE(node, f) = (MSE(node, f) * (f - 1) + norm(eki)^2) / f;

        step_size(node) = 0.004 / (1 + 0.0001 * f);  % Adaptive step size

        % LMS Weight Update
        psi = psi + step_size(node) * eki .* fir_output_nl;
        psi_b(pos_init:pos_end) = psi;

        for m = 1:M
            local_w((node - 1) * M + m) = psi(m);  
        end
    end

    % Apply Metropolis Weighting for Diffusion 
    psi_matrix = reshape(psi_b, M, N);  
    psi_matrix = psi_matrix * metropolis;  
    psi_b = reshape(psi_matrix, N * M, 1);
    
    global_w = mean(local_w, 2);  % Aggregate local weights

    w = global_w;
end

end