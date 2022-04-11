% load first day data
clear all
data = load('t5.2019.05.08_warpedCubes.mat');
training_data = [];
testing_data = [];
testing_labels = [];
labels = [];
% create training set
letters = fieldnames(data);
for i = 1:length(letters)
    if endsWith(letters{i}, '_T')
        tmp = data.(letters{i});
        training_data = [training_data; tmp(:, 1:14)'];
        testing_data = [testing_data; tmp(:, 15:end)'];
        field = letters{i};
        tmp = repmat(string(field(1:end-2)), 1, 14);
        labels = [labels; tmp'];
        testing_labels = [testing_labels; tmp(1:13)'];
    end
end

model = fitcnb(training_data, labels);

predictions = predict(model, testing_data);
accuracy = sum(predictions == testing_labels)/length(testing_labels)

model = fitcdiscr(training_data, labels);

predictions = predict(model, testing_data);
accuracy = sum(predictions == testing_labels)/length(testing_labels)

%%
clear all
data = load('t5.2019.05.08_warpedCubes.mat');
train_num = 17;
test_num = 10;
training_data = zeros(31, train_num, 201, 192);
testing_data = zeros(31, test_num, 201, 192);
letters = fieldnames(data);
n = 1;
for i = 1:length(letters)
    if ~endsWith(letters{i}, '_T')
        tmp = data.(letters{i});
        training_data(n, :, :, :) = tmp(1:train_num, :, :);
        testing_data(n, :, :, :) = tmp(train_num+1:end, :, :);
        n = n+1;
    end
end

training_data(isnan(training_data)) = 0;
testing_data(isnan(testing_data)) = 0;
training_data = squeeze(sum(training_data, 3));
testing_data = squeeze(sum(testing_data, 3));
training_min = min(min(training_data));
testing_min = min(min(testing_data));
for i = 1:192
    training_data(:, :, i) = training_data(:, :, i) - training_min(i);
    testing_data(:, :, i) = testing_data(:, :, i) - testing_min(i);
end

lam = squeeze(mean(training_data, 2));
predictions = zeros([31, test_num]);
results = zeros([31, test_num]);
for i = 1:test_num
    for d = 1:31 % each trial of each letter
        prob_per_letter = zeros([31, 1]);
        for j = 1:31 % probability of each letter (add with neuron)
            for k = 1:192
                p = poisspdf(testing_data(d, i, k), lam(j, k));
                prob_per_letter(j) = prob_per_letter(j) + log10(p);
            end
        end
        [~, predictions(d, i)] = max(prob_per_letter);
        results(d, i) = d;
    end
end

accuracy = sum(sum(results == predictions))/numel(predictions)

%% raw data

clear all
data = load('singleLetters.mat');
train_num = 17;
test_num = 10;
let_num = 30;
training_data = zeros(31, train_num, 201, 192);
testing_data = zeros(31, test_num, 201, 192);
letters = fieldnames(data);
actual = [];
n = 1;
for i = 1:length(letters)
    if numel(size(data.(letters{i})))>=3
        if ~endsWith(letters{i}, '_w')% & ~endsWith(letters{i}, '_v')
            tmp = data.(letters{i});
            training_data(n, :, :, :) = tmp(1:train_num, :, :);
            testing_data(n, :, :, :) = tmp(train_num+1:end, :, :);
            n = n+1;
            tp = letters{i};
            actual = [actual; tp(end-1:end)];
        end
    end
end

training_data(isnan(training_data)) = 0;
testing_data(isnan(testing_data)) = 0;
training_data = squeeze(sum(training_data, 3));
testing_data = squeeze(sum(testing_data, 3));
training_min = min(min(training_data));
testing_min = min(min(testing_data));
for i = 1:192
    training_data(:, :, i) = training_data(:, :, i) - training_min(i);
    testing_data(:, :, i) = testing_data(:, :, i) - testing_min(i);
end

lam = squeeze(mean(training_data, 2));
predictions = zeros([let_num, test_num]);
results = zeros([let_num, test_num]);
for i = 1:test_num
    for d = 1:let_num % each trial of each letter
        prob_per_letter = zeros([let_num, 1]);
        for j = 1:let_num % probability of each letter (add with neuron)
            for k = 1:192
                p = poisspdf(testing_data(d, i, k), lam(j, k));
                prob_per_letter(j) = prob_per_letter(j) + log10(p);
            end
        end
        [~, predictions(d, i)] = max(prob_per_letter);
        results(d, i) = d;
    end
end

accuracy = sum(sum(results == predictions))/numel(predictions)
