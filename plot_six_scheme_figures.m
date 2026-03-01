function plot_six_scheme_figures(base_dir)
% plot_six_scheme_figures
% Generate publication-style figures for six-scheme comparison results.
% Each figure is saved as both .fig and .svg.
%
% Usage:
%   plot_six_scheme_figures
%   plot_six_scheme_figures('D:\Code\dataset\WID\Datasets\transformedData2\six_scheme_eval_20260301_excl_trial02_03')
%
% Font rules:
%   Chinese: FangSong (仿宋)
%   English: Times New Roman
%   Size: 8 pt
%
% Figure width:
%   About 2/3 of A4 width: 14 cm

if nargin < 1 || isempty(base_dir)
    base_dir = 'D:\Code\dataset\WID\Datasets\transformedData2\six_scheme_eval_20260301_excl_trial02_03';
end

manifest_path = fullfile(base_dir, 'manifest_multischemes.csv');
summary_path  = fullfile(base_dir, 'summary_metrics.csv');
gate_path     = fullfile(base_dir, 'gate_results.csv');

assert(exist(manifest_path, 'file') == 2, 'Missing manifest: %s', manifest_path);
assert(exist(summary_path, 'file')  == 2, 'Missing summary: %s', summary_path);
assert(exist(gate_path, 'file')     == 2, 'Missing gate: %s', gate_path);

out_dir = fullfile(base_dir, 'matlab_figures');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

T_manifest = readtable(manifest_path, 'TextType', 'string');
T_summary  = readtable(summary_path, 'TextType', 'string');
T_gate     = readtable(gate_path, 'TextType', 'string'); %#ok<NASGU>

% Ensure numeric columns are numeric
num_cols = ["pos_rmse_60_80_m","speed_rmse_60_80_mps","pos_rmse_global_w_m","speed_rmse_global_w_mps"];
for i = 1:numel(num_cols)
    c = num_cols(i);
    if ~isnumeric(T_summary.(c))
        T_summary.(c) = str2double(T_summary.(c));
    end
end

[T_summary.scheme_key, T_summary.trial] = parse_run_name(T_summary.run_name);
[T_manifest.scheme_key, T_manifest.trial] = parse_run_name(T_manifest.run_name);

scheme_order = ["A","B","C","F","D","E"];
scheme_names_en = containers.Map( ...
    {'A','B','C','F','D','E'}, ...
    {'CT-FGO Main+AllWheel','CT-FGO Main+Rear','CT-FGO Main+Front','CT-FGO MainOnly','KF-GINS Main','Wheel-GINS Rear'} ...
);
scheme_names_cn = containers.Map( ...
    {'A','B','C','F','D','E'}, ...
    {'CT-FGO 主IMU+全轮IMU','CT-FGO 主IMU+后轮IMU','CT-FGO 主IMU+前轮IMU','CT-FGO 仅主IMU','KF-GINS 主IMU','Wheel-GINS 后轮IMU'} ...
);

trials = unique(T_summary.trial, 'stable');

plot_summary_grouped_bars(T_summary, scheme_order, scheme_names_en, scheme_names_cn, out_dir);
plot_summary_boxplots(T_summary, scheme_order, scheme_names_en, scheme_names_cn, out_dir);
plot_trial_scheme_heatmaps(T_summary, trials, scheme_order, scheme_names_en, out_dir);
plot_all_trial_trajectories(T_manifest, trials, scheme_order, scheme_names_en, out_dir);

fprintf('MATLAB figures saved to: %s\n', out_dir);
end


function [scheme_key, trial] = parse_run_name(run_name)
run_name = string(run_name);
n = numel(run_name);
scheme_key = strings(n,1);
trial = strings(n,1);
for i = 1:n
    s = run_name(i);
    t = regexp(s, 'trial\d+$', 'match', 'once');
    trial(i) = string(t);
    if startsWith(s, "schemeA_")
        scheme_key(i) = "A";
    elseif startsWith(s, "schemeB_")
        scheme_key(i) = "B";
    elseif startsWith(s, "schemeC_")
        scheme_key(i) = "C";
    elseif startsWith(s, "schemeF_")
        scheme_key(i) = "F";
    elseif startsWith(s, "schemeD_")
        scheme_key(i) = "D";
    elseif startsWith(s, "schemeE_")
        scheme_key(i) = "E";
    else
        scheme_key(i) = "UNK";
    end
end
end


function plot_summary_grouped_bars(T, scheme_order, map_en, map_cn, out_dir)
vals_pos60 = nan(1, numel(scheme_order));
vals_spd60 = nan(1, numel(scheme_order));
vals_posg  = nan(1, numel(scheme_order));
vals_spdg  = nan(1, numel(scheme_order));
err_pos60  = nan(1, numel(scheme_order));
err_spd60  = nan(1, numel(scheme_order));
err_posg   = nan(1, numel(scheme_order));
err_spdg   = nan(1, numel(scheme_order));

for i = 1:numel(scheme_order)
    k = scheme_order(i);
    I = T.scheme_key == k;
    vals_pos60(i) = mean(T.pos_rmse_60_80_m(I), 'omitnan');
    vals_spd60(i) = mean(T.speed_rmse_60_80_mps(I), 'omitnan');
    vals_posg(i)  = mean(T.pos_rmse_global_w_m(I), 'omitnan');
    vals_spdg(i)  = mean(T.speed_rmse_global_w_mps(I), 'omitnan');

    err_pos60(i) = std(T.pos_rmse_60_80_m(I), 0, 'omitnan');
    err_spd60(i) = std(T.speed_rmse_60_80_mps(I), 0, 'omitnan');
    err_posg(i)  = std(T.pos_rmse_global_w_m(I), 0, 'omitnan');
    err_spdg(i)  = std(T.speed_rmse_global_w_mps(I), 0, 'omitnan');
end

labels = strings(1, numel(scheme_order));
for i = 1:numel(scheme_order)
    labels(i) = map_en(char(scheme_order(i)));
end

% Figure 1: 60-80 metrics
f = make_fig_cm(14, 8);
ax = axes(f);
hold(ax, 'on');
x = 1:numel(scheme_order);
w = 0.38;
b1 = bar(ax, x - w/2, vals_pos60, w, 'FaceColor', [0.23 0.49 0.74]);
b2 = bar(ax, x + w/2, vals_spd60, w, 'FaceColor', [0.85 0.33 0.10]);
errorbar(ax, x - w/2, vals_pos60, err_pos60, 'k.', 'LineWidth', 0.8);
errorbar(ax, x + w/2, vals_spd60, err_spd60, 'k.', 'LineWidth', 0.8);
set(ax, 'XTick', x, 'XTickLabel', labels);
xtickangle(ax, 20);
ylabel(ax, 'RMSE');
legend(ax, [b1 b2], {'Pos2D 60-80 (m)', 'Speed2D 60-80 (m/s)'}, 'Location', 'northwest');
title(ax, '\fontname{仿宋}各方案60-80s指标均值\fontname{Times New Roman}  Mean Metrics');
apply_axis_style(ax);
save_fig_pair(f, out_dir, 'summary_60_80_mean_std');

% Figure 2: global weighted metrics
f = make_fig_cm(14, 8);
ax = axes(f);
hold(ax, 'on');
b1 = bar(ax, x - w/2, vals_posg, w, 'FaceColor', [0.23 0.49 0.74]);
b2 = bar(ax, x + w/2, vals_spdg, w, 'FaceColor', [0.85 0.33 0.10]);
errorbar(ax, x - w/2, vals_posg, err_posg, 'k.', 'LineWidth', 0.8);
errorbar(ax, x + w/2, vals_spdg, err_spdg, 'k.', 'LineWidth', 0.8);
set(ax, 'XTick', x, 'XTickLabel', labels);
xtickangle(ax, 20);
ylabel(ax, 'RMSE');
legend(ax, [b1 b2], {'Pos2D GlobalW (m)', 'Speed2D GlobalW (m/s)'}, 'Location', 'northwest');
title(ax, '\fontname{仿宋}各方案全局加权指标均值\fontname{Times New Roman}  Mean Global Weighted Metrics');
apply_axis_style(ax);
save_fig_pair(f, out_dir, 'summary_global_weighted_mean_std');

% Figure 3: chinese label summary table-like chart (pos60 only)
f = make_fig_cm(14, 8);
ax = axes(f);
bar(ax, x, vals_pos60, 0.6, 'FaceColor', [0.20 0.62 0.56]);
hold(ax, 'on');
errorbar(ax, x, vals_pos60, err_pos60, 'k.', 'LineWidth', 0.8);
cn_labels = strings(1, numel(scheme_order));
for i = 1:numel(scheme_order)
    cn_labels(i) = map_cn(char(scheme_order(i)));
end
set(ax, 'XTick', x, 'XTickLabel', cn_labels);
xtickangle(ax, 20);
ylabel(ax, '\fontname{Times New Roman}Pos2D RMSE (m)');
title(ax, '\fontname{仿宋}各方案位置误差统计（60-80s）');
apply_axis_style(ax);
save_fig_pair(f, out_dir, 'summary_pos60_cn');
end


function plot_summary_boxplots(T, scheme_order, map_en, map_cn, out_dir) %#ok<INUSD>
x_labels = strings(1, numel(scheme_order));
for i = 1:numel(scheme_order)
    x_labels(i) = map_en(char(scheme_order(i)));
end

% 60-80 position boxplot
f = make_fig_cm(14, 8);
ax = axes(f);
hold(ax, 'on');
groups = categorical(T.scheme_key, scheme_order, scheme_order);
boxchart(ax, groups, T.pos_rmse_60_80_m, 'BoxFaceColor', [0.23 0.49 0.74]);
ylabel(ax, 'Pos2D RMSE 60-80 (m)');
title(ax, '\fontname{仿宋}各方案位置误差分布（按轨迹）\fontname{Times New Roman}  Boxplot');
set(ax, 'XTickLabel', x_labels);
xtickangle(ax, 20);
apply_axis_style(ax);
save_fig_pair(f, out_dir, 'boxplot_pos60_by_scheme');

% 60-80 speed boxplot
f = make_fig_cm(14, 8);
ax = axes(f);
hold(ax, 'on');
boxchart(ax, groups, T.speed_rmse_60_80_mps, 'BoxFaceColor', [0.85 0.33 0.10]);
ylabel(ax, 'Speed2D RMSE 60-80 (m/s)');
title(ax, '\fontname{仿宋}各方案速度误差分布（按轨迹）\fontname{Times New Roman}  Boxplot');
set(ax, 'XTickLabel', x_labels);
xtickangle(ax, 20);
apply_axis_style(ax);
save_fig_pair(f, out_dir, 'boxplot_speed60_by_scheme');
end


function plot_trial_scheme_heatmaps(T, trials, scheme_order, map_en, out_dir)
nT = numel(trials);
nS = numel(scheme_order);
M_pos = nan(nT, nS);
M_spd = nan(nT, nS);
for i = 1:nT
    for j = 1:nS
        I = T.trial == trials(i) & T.scheme_key == scheme_order(j);
        if any(I)
            M_pos(i,j) = T.pos_rmse_60_80_m(find(I,1,'first'));
            M_spd(i,j) = T.speed_rmse_60_80_mps(find(I,1,'first'));
        end
    end
end

scheme_labels = strings(1, nS);
for j = 1:nS
    scheme_labels(j) = map_en(char(scheme_order(j)));
end

f = make_fig_cm(14, 9);
ax = axes(f);
imagesc(ax, M_pos);
colormap(ax, parula(256));
cb = colorbar(ax);
cb.Label.String = 'Pos2D RMSE 60-80 (m)';
set(ax, 'XTick', 1:nS, 'XTickLabel', scheme_labels, 'YTick', 1:nT, 'YTickLabel', trials);
xtickangle(ax, 20);
title(ax, '\fontname{仿宋}位置误差热力图（轨迹×方案）\fontname{Times New Roman}');
apply_axis_style(ax);
save_fig_pair(f, out_dir, 'heatmap_pos60_trial_scheme');

f = make_fig_cm(14, 9);
ax = axes(f);
imagesc(ax, M_spd);
colormap(ax, turbo(256));
cb = colorbar(ax);
cb.Label.String = 'Speed2D RMSE 60-80 (m/s)';
set(ax, 'XTick', 1:nS, 'XTickLabel', scheme_labels, 'YTick', 1:nT, 'YTickLabel', trials);
xtickangle(ax, 20);
title(ax, '\fontname{仿宋}速度误差热力图（轨迹×方案）\fontname{Times New Roman}');
apply_axis_style(ax);
save_fig_pair(f, out_dir, 'heatmap_speed60_trial_scheme');
end


function plot_all_trial_trajectories(T_manifest, trials, scheme_order, map_en, out_dir)
traj_dir = fullfile(out_dir, 'traj_plots');
if ~exist(traj_dir, 'dir')
    mkdir(traj_dir);
end

for it = 1:numel(trials)
    tr = trials(it);
    Itrial = T_manifest.trial == tr;
    Tt = T_manifest(Itrial, :);
    if isempty(Tt)
        continue;
    end

    truth_path = char(Tt.truth_path(1));
    if exist(truth_path, 'file') ~= 2
        continue;
    end
    truth = readmatrix(truth_path);
    if size(truth,2) < 4
        continue;
    end

    t0 = truth(1,1);
    lat0 = truth(1,2);
    lon0 = truth(1,3);
    h0 = truth(1,4);
    [E_truth, N_truth, U_truth] = llh_to_enu(truth(:,2), truth(:,3), truth(:,4), lat0, lon0, h0);

    % 2D trajectory plot per trial
    f = make_fig_cm(14, 8);
    ax = axes(f);
    hold(ax, 'on');
    plot(ax, E_truth, N_truth, 'k-', 'LineWidth', 1.0, 'DisplayName', 'Truth');

    for is = 1:numel(scheme_order)
        k = scheme_order(is);
        Ir = Tt.scheme_key == k;
        if ~any(Ir)
            continue;
        end
        rp = char(Tt.result_path(find(Ir,1,'first')));
        if exist(rp, 'file') ~= 2
            continue;
        end
        R = readmatrix(rp);
        if size(R,2) < 4
            continue;
        end
        [E, N, ~] = llh_to_enu(R(:,2), R(:,3), R(:,4), lat0, lon0, h0);
        plot(ax, E, N, 'LineWidth', 0.8, 'DisplayName', map_en(char(k)));
    end

    axis(ax, 'equal');
    xlabel(ax, 'East (m)');
    ylabel(ax, 'North (m)');
    title(ax, ['\fontname{仿宋}轨迹对比\fontname{Times New Roman}  ', char(tr)]);
    legend(ax, 'Location', 'bestoutside');
    apply_axis_style(ax);
    save_fig_pair(f, traj_dir, ['traj2d_' char(tr)]);

    % ENU displacement (3 axes) per trial
    f = make_fig_cm(14, 11);
    t_truth = truth(:,1) - t0;
    axs(1) = subplot(3,1,1); %#ok<AGROW>
    plot(axs(1), t_truth, E_truth, 'k-', 'LineWidth', 1.0); hold(axs(1), 'on');
    ylabel(axs(1), 'E (m)');
    title(axs(1), ['\fontname{仿宋}ENU位移对比\fontname{Times New Roman}  ', char(tr)]);

    axs(2) = subplot(3,1,2);
    plot(axs(2), t_truth, N_truth, 'k-', 'LineWidth', 1.0); hold(axs(2), 'on');
    ylabel(axs(2), 'N (m)');

    axs(3) = subplot(3,1,3);
    plot(axs(3), t_truth, U_truth, 'k-', 'LineWidth', 1.0); hold(axs(3), 'on');
    ylabel(axs(3), 'U (m)');
    xlabel(axs(3), 'Time (s)');

    for is = 1:numel(scheme_order)
        k = scheme_order(is);
        Ir = Tt.scheme_key == k;
        if ~any(Ir)
            continue;
        end
        rp = char(Tt.result_path(find(Ir,1,'first')));
        if exist(rp, 'file') ~= 2
            continue;
        end
        R = readmatrix(rp);
        if size(R,2) < 4
            continue;
        end
        trr = R(:,1) - t0;
        [E, N, U] = llh_to_enu(R(:,2), R(:,3), R(:,4), lat0, lon0, h0);
        plot(axs(1), trr, E, 'LineWidth', 0.7, 'DisplayName', map_en(char(k)));
        plot(axs(2), trr, N, 'LineWidth', 0.7, 'DisplayName', map_en(char(k)));
        plot(axs(3), trr, U, 'LineWidth', 0.7, 'DisplayName', map_en(char(k)));
    end
    legend(axs(1), 'Location', 'bestoutside');
    for ia = 1:3
        apply_axis_style(axs(ia));
    end
    save_fig_pair(f, traj_dir, ['enu_' char(tr)]);
end
end


function [E,N,U] = llh_to_enu(lat_deg, lon_deg, h_m, lat0_deg, lon0_deg, h0_m)
R = 6378137.0;
dlat = deg2rad(lat_deg - lat0_deg);
dlon = deg2rad(lon_deg - lon0_deg);
N = dlat .* R;
E = dlon .* R .* cosd(lat0_deg);
U = h_m - h0_m;
end


function f = make_fig_cm(w_cm, h_cm)
f = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 w_cm h_cm], 'PaperPositionMode', 'auto');
end


function apply_axis_style(ax)
set(ax, 'FontName', 'Times New Roman', 'FontSize', 8, 'LineWidth', 0.6, 'Box', 'on');
grid(ax, 'on');
ax.GridAlpha = 0.25;
% Set Chinese font for title if any CJK chars exist.
if contains(ax.Title.String, char([19968 40857])) || contains(ax.Title.String, '\fontname{仿宋}')
    ax.Title.FontName = '仿宋';
else
    ax.Title.FontName = 'Times New Roman';
end
ax.Title.FontSize = 8;
end


function save_fig_pair(f, out_dir, base_name)
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
fig_path = fullfile(out_dir, [base_name '.fig']);
svg_path = fullfile(out_dir, [base_name '.svg']);
savefig(f, fig_path);
exportgraphics(f, svg_path, 'ContentType', 'vector');
close(f);
end
