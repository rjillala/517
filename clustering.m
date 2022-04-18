data = load('data/handwritingBCIData/Datasets/t5.2019.05.08/singleLetters.mat');
train_num = 23;
training_data = zeros(let_num, train_num, 101, 192);
letters = fieldnames(data);
n = 1;
for i = 1:let_num
    if numel(size(data.(letters{i})))>=3
        tmp = data.(letters{i});
        training_data(n, :, :, :) = tmp(1:train_num, 51:151, :);
        n = n+1;
    end
end
training_data = squeeze(sum(training_data, 3));
training = zeros([train_num*let_num, 192]);
training_class = zeros([train_num*let_num, 1]);
n = 1;
for i = 1:let_num
    for j = 1:train_num
        training(n, :) = squeeze(training_data(i, j, :));
        training_class(n) = i;
        n = n + 1;
    end
end
training_data = training;

[coeffs, scores] = pca(training_data);
training_data = scores(:, 1:44);

%% plot clusters
figure
X = training_data;
assignments_1 = zeros([height(X), 1])+1;
for l = 1
    % set K
    K = 31;

    % set K random centroids
    indices = zeros([K, 1]);
    while unique(indices) < K
        indices = randi(height(X), [K, 1]);
    end
    centroids = X(indices, :);

    % loop until convergence
    assignments = zeros([height(X), 1])+1;
    pre_assignments = zeros([height(X), 1]);
    n = 1;
    while sum(pre_assignments ~= assignments) > 0
        n = n + 1;
        pre_assignments = assignments;

        % get distance from centroid
        distances = zeros([height(X), K]);
        for i = 1:height(X)
            for j = 1:K
                distances(i, j) = sqrt((X(i, 1)-centroids(j, 1))^2 + (X(i, 2)-centroids(j, 2))^2);
            end
        end

        % assign points to a cluster
        [~, assignments] = min(distances, [], 2);

        % recalculate centroids
        for i = 1:K
            centroids(i, :) = mean(X(assignments==i, :));
        end
    end
    if l == 1
        assignments_1 = assignments;
    end
    % plot clusters
    gscatter(X(:, 1), X(:, 2), assignments, [], [], 10)
    title(strcat("PCA Converted Spikes Clustered Using K-Means: Run #", string(l)))
    xlabel("Principal Component 1")
    ylabel("Principal Component 2")
    legg = strings([1, K]);
    for i = 1:K
        legg(i) = strcat("Cluster ", string(i));
    end
    legend(legg)
end