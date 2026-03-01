function export_hailaer_mat_to_txt()
% Batch-convert hailaer inertial datasets (.mat) to high-precision text files
% for CT_FGO and KF-GINS (main IMU + GNSS only).

root_dir = 'D:/Code/dataset/hailaer/inertail';
out_root = fullfile(root_dir, 'ctd_kf_compare_20260301');
if ~exist(out_root, 'dir')
    mkdir(out_root);
end

files = [ ...
    "D:/Code/dataset/hailaer/inertail/20260202_144409/transformed1cut1/aligned_imu_gnss_rtk.mat"
    "D:/Code/dataset/hailaer/inertail/20260202_144409/transformed1cut2/aligned_imu_gnss_rtk.mat"
    "D:/Code/dataset/hailaer/inertail/20260202_144409/transformed1cut3/aligned_imu_gnss_rtk.mat"
    "D:/Code/dataset/hailaer/inertail/20260202_150640/transformed1cut1/aligned_imu_gnss_rtk.mat"
    "D:/Code/dataset/hailaer/inertail/20260202_150640/transformed1cut2/aligned_imu_gnss_rtk.mat"
    "D:/Code/dataset/hailaer/inertail/20260202_150640/transformed1cut3/aligned_imu_gnss_rtk.mat"
    "D:/Code/dataset/hailaer/inertail/20260202_150640/transformed1cut4/aligned_imu_gnss_rtk.mat" ...
];

manifest_path = fullfile(out_root, 'dataset_manifest.csv');
mfid = fopen(manifest_path, 'w');
fprintf(mfid, 'dataset_id,source_mat,output_dir,imu_file,gnss_file,rtk_file,t_start,t_end\n');

for i = 1:numel(files)
    mat_path = char(files(i));
    [parent_dir, ~, ~] = fileparts(mat_path);
    rel_id = strrep(parent_dir, 'D:/Code/dataset/hailaer/inertail/', '');
    dataset_id = strrep(rel_id, '/', '__');
    out_dir = fullfile(out_root, dataset_id);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    S = load(mat_path);
    p = S.params;
    imu = S.out.imu;
    gnss = S.out.gnss;
    rtk = S.out.rtk;

    t_imu = imu(:, p.imu_cols.time1) * p.imu_time_scale;
    gx = imu(:, p.imu_cols.gx);
    gy = imu(:, p.imu_cols.gy);
    gz = imu(:, p.imu_cols.gz);
    ax = imu(:, p.imu_cols.ax);
    ay = imu(:, p.imu_cols.ay);
    az = imu(:, p.imu_cols.az);

    dt = [median(diff(t_imu)); diff(t_imu)];
    dt(~isfinite(dt) | dt <= 0) = median(dt(dt > 0));

    % Convert gyro rates to increments in radians.
    if strcmpi(p.imu_gyro_unit, 'degph')
        scale_g = pi / 180 / 3600;
    elseif strcmpi(p.imu_gyro_unit, 'degps')
        scale_g = pi / 180;
    elseif strcmpi(p.imu_gyro_unit, 'radps')
        scale_g = 1;
    else
        error('Unsupported imu_gyro_unit: %s', p.imu_gyro_unit);
    end
    dtheta = [gx, gy, gz] * scale_g .* dt;

    if strcmpi(p.imu_acc_unit, 'mps2')
        acc_scale = 1;
    else
        error('Unsupported imu_acc_unit: %s', p.imu_acc_unit);
    end
    dvel = [ax, ay, az] * acc_scale .* dt;

    imu_out = [t_imu, dtheta, dvel];

    t_gnss = gnss(:, p.gnss_cols.time1) * p.gnss_time_scale;
    lat_gnss = gnss(:, p.gnss_cols.lat);
    lon_gnss = gnss(:, p.gnss_cols.lon);
    h_gnss = gnss(:, p.gnss_cols.h);
    gnss_std = [ones(size(t_gnss)), ones(size(t_gnss)), 2 * ones(size(t_gnss))];
    gnss_out = [t_gnss, lat_gnss, lon_gnss, h_gnss, gnss_std];

    t_rtk = rtk(:, p.rtk_cols.time1) * p.rtk_time_scale;
    lat_rtk = rtk(:, p.rtk_cols.lat);
    lon_rtk = rtk(:, p.rtk_cols.lon);
    h_rtk = rtk(:, p.rtk_cols.h);
    rtk_out = [t_rtk, lat_rtk, lon_rtk, h_rtk];

    imu_file = fullfile(out_dir, 'Body_IMU.txt');
    gnss_file = fullfile(out_dir, 'GNSS_low.txt');
    rtk_file = fullfile(out_dir, 'RTK_truth.txt');

    writematrix(imu_out, imu_file, 'Delimiter', ' ', 'FileType', 'text');
    writematrix(gnss_out, gnss_file, 'Delimiter', ' ', 'FileType', 'text');
    writematrix(rtk_out, rtk_file, 'Delimiter', ' ', 'FileType', 'text');

    % Re-write with explicit scientific notation for precision consistency.
    rewrite_with_sci(imu_file, imu_out);
    rewrite_with_sci(gnss_file, gnss_out);
    rewrite_with_sci(rtk_file, rtk_out);

    t_start = max([t_imu(1), t_gnss(1), t_rtk(1)]);
    t_end = min([t_imu(end), t_gnss(end), t_rtk(end)]);

    fprintf(mfid, '%s,%s,%s,%s,%s,%s,%.9f,%.9f\n', ...
        dataset_id, mat_path, out_dir, imu_file, gnss_file, rtk_file, t_start, t_end);
end

fclose(mfid);
fprintf('Export complete. Manifest: %s\n', manifest_path);
end

function rewrite_with_sci(path, M)
fid = fopen(path, 'w');
if fid < 0
    error('Cannot open %s for writing', path);
end
ncol = size(M, 2);
fmt_cells = repmat({'%.16e'}, 1, ncol);
fmt = strjoin(fmt_cells, ' ');
fmt = [fmt, '\n'];
for i = 1:size(M, 1)
    fprintf(fid, fmt, M(i, :));
end
fclose(fid);
end
