function [phi, I_m_phi, phi_inv, I_f_phi_inv,toc_phi, toc_phi_inv] = VPreg_gpu(I_m, I_f)
% GPU-first image registration: moves major arrays to GPU and gathers outputs at end.
tic;
J_thr = 0.55;
regrid_bd = 8;
% Keep original copies for final CPU-side interpolation if needed
I_m_orig = gpuArray(single(I_m));
I_f_orig = gpuArray(single(I_f));

% sizes
[nxm, nym, nzm] = size(I_m_orig);
[nxf, nyf, nzf] = size(I_f_orig);

% Move images to GPU and convert to single for memory savings
I_m = voxel2zscore(I_m);
I_f = voxel2zscore(I_f);

alternating = regrid_bd - 2;

%% Image registration parameters (unchanged)
stage = 0;
stagnate = 1;
ts_initseq = 3.3e-3;
ts_init = 3.3e-4;
ts = ts_initseq;
ts_rtol = 1e-6;
modo = ['seq', '-mod'];
lr_ctlseq = 0.1;
lr_ctl = 0.01;
ts_up = 1 + lr_ctlseq;
ts_dn = 1 - lr_ctlseq;
imax = 1.5e2;
rtol = 7e-1;
rr = 1;

% Step-1: grid and variables (on GPU)
grid_size = [nxf, nyf, nzf];
f = gpuArray(single(ones(grid_size)));
g = gpuArray(single(zeros([grid_size, 3])));

% Compute initial ssd on GPU
ssd_old = sum((I_m - I_f).^2, "all");
ssd_initial = ssd_old;
ssd_initial_temp = ssd_old;

% mutual information: if mutualInfo expects cpu, gather inputs; assume it can run on gpu
try
    mi = mutualInfo(I_m, I_f);
catch
    mi = mutualInfo(gather(I_m), gather(I_f));
end

Better = 1;
true_better = 0;
ti = 0;
ei = 0;

% coordinate grids (gpu)
[X, Y, Z] = ndgrid(1:nxf, 1:nyf, 1:nzf);
X = gpuArray(single(X));
Y = gpuArray(single(Y));
Z = gpuArray(single(Z));

phi_x = X;
phi_y = Y;
phi_z = Z;

phi_x_interm = X;
phi_y_interm = Y;
phi_z_interm = Z;

I_m_local = I_m;
JacD_below = 1;

alli = 0;

Nx = nxm; Ny = nym; Nz = nzm;
tx = 4 * sin((pi/(2*(Nx+1))) * (1:Nx)).^2;
ty = 4 * sin((pi/(2*(Ny+1))) * (1:Ny)).^2;
tz = 4 * sin((pi/(2*(Nz+1))) * (1:Nz)).^2;
G = gpuArray(single(reshape(tx,[],1,1) + reshape(ty,1,[],1) + reshape(tz,1,1,[])));

lap_phi = gpuArray(single(zeros([size(I_m), 3])));
lap_B = gpuArray(single(zeros([size(I_m), 3])));
del_g = gpuArray(single(zeros([size(I_m), 3])));
fg = gpuArray(single(zeros([size(f), 3])));
c_p = 5;

lll = 0;

while (lll <= regrid_bd)
    while  (rr > rtol) && (ts > ts_rtol) && (stage < stagnate) && (ei < imax) && (JacD_below >= J_thr)
        ti = ti + 1;
        alli = alli + 1;
        if Better
            ei = ei + 1;
            if lll < alternating
                
                % compute JD and curl (assumed GPU-capable)
                [fi, gi1, gi2, gi3] = compute_JD_and_Curl3D(phi_x, phi_y, phi_z);
                [fi_y, fi_x, fi_z] = gradient(fi);
                [~, curl_gi1, curl_gi2, curl_gi3] = compute_JD_and_Curl3D(gi1, gi2, gi3);

                % warp moving image on GPU (interpn supports gpuArray)
                I_m_temp = interpn(X, Y, Z, I_m_local, phi_x, phi_y, phi_z, 'linear');
                Img_diff = I_m_temp - I_f;
                [I_m_y, I_m_x, I_m_z] = gradient(I_m_temp);

                lap_phi(:,:,:,1) = Img_diff .* I_m_x + c_p*(fi_x - curl_gi1);
                lap_phi(:,:,:,2) = Img_diff .* I_m_y + c_p*(fi_y - curl_gi2);
                lap_phi(:,:,:,3) = Img_diff .* I_m_z + c_p*(fi_z - curl_gi3);

                % Solve Poisson on GPU (assumed GPU-capable)
                U = gpuPoissonSolver4D(-lap_phi, G);
            else
                I_m_temp = interpn(X, Y, Z, I_m_local, phi_x, phi_y, phi_z, 'linear');
                Img_diff = I_m_temp - I_f;
                [I_m_y, I_m_x, I_m_z] = gradient(I_m_temp);

                lap_B(:,:,:,1) = Img_diff .* I_m_x;
                lap_B(:,:,:,2) = Img_diff .* I_m_y;
                lap_B(:,:,:,3) = Img_diff .* I_m_z;

                B = gpuPoissonSolver4D(lap_B, G);

                [B1_x2,  B1_x1,  B1_x3] = gradient(B(:,:,:,1));
                [B2_x2,  B2_x1,  B2_x3] = gradient(B(:,:,:,2));
                [B3_x2,  B3_x1,  B3_x3] = gradient(B(:,:,:,3));

                % del_f = imgaussfilt3(gather(- f .* (B1_x1 + B2_x2 + B3_x3)), 3); % imgaussfilt3 lacks gpu support: do on CPU
                % del_f = gpuArray(single(del_f));
                del_f = - f .* (B1_x1 + B2_x2 + B3_x3);              
                del_g(:,:,:,1) = - B3_x2 + B2_x3;
                del_g(:,:,:,2) =   B3_x1 - B1_x3;
                del_g(:,:,:,3) = - B2_x1 + B1_x2;
            end
        end

        % apply update
        if lll < alternating
            % we assume U is on GPU and has size [nxf,nyf,nzf,3]
            phi_x = X + ts * matrixpad3D(squeeze(U(2:nxf-1,2:nyf-1,2:nzf-1,1)), 0);
            phi_y = Y + ts * matrixpad3D(squeeze(U(2:nxf-1,2:nyf-1,2:nzf-1,2)), 0);
            phi_z = Z + ts * matrixpad3D(squeeze(U(2:nxf-1,2:nyf-1,2:nzf-1,3)), 0);
        else
            f_n = f - del_f * ts;
            g_n = g - del_g * ts;

            [f_x2, f_x1, f_x3] = gradient(f_n);
            [g1_x2, ~, g1_x3] = gradient(g_n(:,:,:,1));
            [~, g2_x1, g2_x3] = gradient(g_n(:,:,:,2));
            [g3_x2, g3_x1, ~] = gradient(g_n(:,:,:,3));

            fg1 = f_x1 - g3_x2 + g2_x3;
            fg2 = f_x2 + g3_x1 - g1_x3;
            fg3 = f_x3 - g2_x1 + g1_x2;
            
            fg(:,:,:,1) = fg1;
            fg(:,:,:,2) = fg2;
            fg(:,:,:,3) = fg3;

            U = gpuPoissonSolver4D(fg, G);

            phi_x = X + matrixpad3D(squeeze(U(2:nxf-1,2:nyf-1,2:nzf-1,1)), 0);
            phi_y = Y + matrixpad3D(squeeze(U(2:nxf-1,2:nyf-1,2:nzf-1,2)), 0);
            phi_z = Z + matrixpad3D(squeeze(U(2:nxf-1,2:nyf-1,2:nzf-1,3)), 0);
        end

        % Evaluate Jacobian and metrics
        [JacD2, ~] = compute_JD_and_Curl3D(phi_x, phi_y, phi_z);
        JacD_below = min(JacD2, [], "all");
        I_m_phi = interpn(X, Y, Z, I_m_local, phi_x, phi_y, phi_z, 'linear');
        try
            mi_fwd = mutualInfo(I_m_phi, I_f);
        catch
            mi_fwd = mutualInfo(gather(I_m_phi), gather(I_f));
        end
        mi_inc = (mi_fwd - mi) / mi;
        ssd = sum((I_m_phi - I_f).^2, "all");
        rr = ssd / ssd_initial_temp;
        r = ssd / ssd_initial;

        if ssd < ssd_old
            stage = 0;
            I_m_better = I_m_phi;
            ts = ts * ts_up;
            ssd_old = ssd;
            Better = 1;
            if lll >= alternating
                f = f_n;
                g = g_n;
                modo = ['f&g', '-mod'];
            end
            phi_x_better = phi_x;
            phi_y_better = phi_y;
            phi_z_better = phi_z;
            true_better = 1;
        else
            Better = 0;
            ts = ts * ts_dn;
            stage = stage + 1;
        end

    end
    disp([modo,' c#:',num2str(lll),' e/t:',num2str(ei),'/',num2str(ti),' s:',num2str(ts),' r/rr:',num2str(r),'/',num2str(rr),' Jmin:',num2str(JacD_below),' rMI:', num2str(mi_inc*100)]);

    JacD_below = 1;
    if true_better > 0
        I_m_local = I_m_better;
        phi_x_interm = interpn(X, Y, Z, phi_x_interm, phi_x_better, phi_y_better, phi_z_better, 'linear');
        phi_y_interm = interpn(X, Y, Z, phi_y_interm, phi_x_better, phi_y_better, phi_z_better, 'linear');
        phi_z_interm = interpn(X, Y, Z, phi_z_interm, phi_x_better, phi_y_better, phi_z_better, 'linear');

        I_m_phi_true = interpn(X, Y, Z, I_m, phi_x_interm, phi_y_interm, phi_z_interm, 'linear');
        ssd_old = sum((I_m_phi_true - I_f).^2, "all");
        r_fwd = ssd_old / ssd_initial;
        ssd_initial_temp = ssd_old;
        JD_whole = compute_JD3D(phi_x_interm, phi_y_interm, phi_z_interm);
        JD_min_whole = min(JD_whole, [], "all");
        true_better = 0;
        display([' r_fwd:', num2str(gather(r_fwd)),' J_min_comp:', num2str(gather(JD_min_whole))]);
    end

    phi_x = X;
    phi_y = Y;
    phi_z = Z;
    lll = lll + 1;
    stage = 0;
    Better = 1;
    rr = 1;
    ei = 0;
    ti = 0;

    if lll < alternating
        ts = ts_initseq;
        ts_up = 1 + lr_ctlseq;
        ts_dn = 1 - lr_ctlseq;
    else
        f = gpuArray(single(ones(grid_size)));
        g = gpuArray(single(zeros([grid_size, 3])));
        ts = ts_init;
        ts_up = 1 + lr_ctl;
        ts_dn = 1 - lr_ctl;
        rtol = 0.75;
        J_thr = 0.688;
        stagnate = stagnate + 1;
    end
    imax = imax - 10;
end

phi = zeros(nxf, nyf, nzf, 3, 'single');
phi(:,:,:,1) = gather(phi_x_interm);
phi(:,:,:,2) = gather(phi_y_interm);
phi(:,:,:,3) = gather(phi_z_interm);

I_m_phi = interpn(gather(X), gather(Y), gather(Z), gather(I_m_orig), phi(:,:,:,1), phi(:,:,:,2), phi(:,:,:,3), 'makima');
I_m_phi = imresize3(I_m_phi, [nxm nym nzm]);
toc_phi = toc

tic;
[~,~,~,phi_x_inv,phi_y_inv,phi_z_inv] = VP_LM_gpu(nxf, nyf, nzf, phi_x_interm, phi_y_interm, phi_z_interm, X, Y, Z);
toc_phi_inv = toc

phi_inv = zeros(nxf, nyf, nzf, 3, 'single');
phi_inv(:,:,:,1) = phi_x_inv;
phi_inv(:,:,:,2) = phi_y_inv;
phi_inv(:,:,:,3) = phi_z_inv;

I_f_phi_inv = interpn(gather(X), gather(Y), gather(Z), gather(I_f_orig), phi_inv(:,:,:,1), phi_inv(:,:,:,2), phi_inv(:,:,:,3), 'makima');
end

function img_z=voxel2zscore(img)
img_mean=mean(img,"all");
img_std=std(img,0,"all");
scale_is = 1/img_std;
img_z=(img-img_mean)*scale_is;
end

function [JD_phi, CV_phi1, CV_phi2, CV_phi3]=compute_JD_and_Curl3D(Phi1,Phi2,Phi3)
    [Phi1y,Phi1x,Phi1z]=gradient(Phi1);
    [Phi2y,Phi2x,Phi2z]=gradient(Phi2);
    [Phi3y,Phi3x,Phi3z]=gradient(Phi3);
    
    
    JD_phi=Phi1x.*(Phi2y.*Phi3z-Phi2z.*Phi3y)...
        -Phi2x.*(Phi1y.*Phi3z-Phi1z.*Phi3y)...
        +Phi3x.*(Phi1y.*Phi2z-Phi1z.*Phi2y);
    
    CV_phi1=Phi3y-Phi2z;
    CV_phi2=Phi1z-Phi3x;
    CV_phi3=Phi2x-Phi1y;
end

function [B] = matrixpad3D(A,x)
    % pad A with boundary x 
    [m,n,l]=size(A);
    B=ones(m+2,n+2,l+2)*x;
    B(2:m+1,2:n+1,2:l+1)=A;
end

function U = gpuPoissonSolver4D(F_4D, G)
    % F_4D: [Nx, Ny, Nz, 3] gpuArray (single)
    % G: [Nx, Ny, Nz] precomputed denominator
    
    % Forward 3D DST: Sequential 1D transforms
    U = gpu_dst_exact(F_4D, 1);
    U = gpu_dst_exact(U, 2);
    U = gpu_dst_exact(U, 3);
    
    % Solve in Frequency Domain
    U = U ./ G;
    
    % Inverse 3D DST (IDST-I is identical to DST-I with a scaling factor)
    U = gpu_dst_exact(U, 1);
    U = gpu_dst_exact(U, 2);
    U = gpu_dst_exact(U, 3);
    
    % Final scaling to match your 'idst' logic exactly
    [Nx, Ny, Nz, ~] = size(F_4D);
    scale = (2/(Nx+1)) * (2/(Ny+1)) * (2/(Nz+1));
    U = U * (scale * 0.125); % 0.125 compensates for FFT mirroring
end

function out = gpu_dst_exact(data, dim)
n = size(data, dim);
N = 2*n + 2;

% Create padded array for the active dimension
padSz = size(data);
padSz(dim) = N;
pad = gpuArray.zeros(padSz, 'single');

% DST-I Symmetry: [0, data, 0, -flip(data)]
idx1 = repmat({':'}, 1, 4); 
idx1{dim} = 2:n+1;
pad(idx1{:}) = data;

idx2 = repmat({':'}, 1, 4);
idx2{dim} = n+3:N;
pad(idx2{:}) = -flip(data, dim);

% FFT along the specific dimension
res = fft(pad, [], dim);

% Return the imaginary part of the first n points
out = -imag(res(idx1{:}));
end

