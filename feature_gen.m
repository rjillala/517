% load data 
% data = load('t5.2019.05.08_warpedCubes');
dir = pwd;
data = load(strcat(dir, '/Datasets/t5.2019.05.08/singleLetters.mat'));
warped = load(strcat(dir, '/RNNTrainingSteps/Step1_TimeWarping/t5.2019.05.08_warpedCubes.mat'));
velocity = load(strcat(dir, '/Datasets/computerMouseTemplates.mat'));

%% Calculate line length for each trial 
clear ll

letters = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p', ...
    'q','r','s','t','u','v','w','x','y','z','greaterThan', ... 'doNothing', ...
    'comma','apostrophe','tilde','questionMark'};
num_l = numel(letters);

for letter = 1:num_l
    for k = 1:192
        for i = 1:27
            data_name = strcat('neuralActivityCube_', letters{letter});
            warped_name = strcat(letters{letter});

            d = data.(data_name)(i,:,k);
            crossing(i,k,letter) = sum(d);
                     
            w = warped.(warped_name)(i,:,k);
            w(isnan(w)) = 0;
            ll(i,k,letter) = sum(abs(diff(w)));
            

        end
    end
end

%%
data1 = [];
velocity1 = [];

for letter = 1:num_l
    [m,~] = size(velocity.(letters{letter}));
    temp = [velocity.(letters{letter}); zeros(100-m, 2)];
%     for i = 1:27
%         velocity1 = [velocity1; temp(1:100,:)];
%     end

    velocity1 = [velocity1; temp(1:100,:)];
    
    temp_data1 = [];
    for k = 1:192
%         data_name = strcat('neuralActivityCube_', letters{letter});
%         temp_data = data.(data_name)(:,51:150,k); 
        data_name = letters{letter};
        temp_data = warped.(data_name)(:,51:150,k); 
        temp_data(isnan(temp_data)) = 0;
        temp_data1 = [temp_data1, mean(temp_data)'];
%         temp_data1 = [temp_data1, temp_data(:)];
    end

    data1 = [data1; temp_data1];
    
end

%% Naive Bayes
% load data and separate into train and test datasets 
clear metric

for k = 1:num_l
%     metric(:,:,k) = [crossing(:,51:150,k), ll(:,51:150,k)];
    metric(:,:,k) = [crossing(:,51:150,k)];
end

train = metric(1:13,:,:);
test = metric(14:end,:,:);
[~,n,r] = size(metric);

% calculate mean of each feature-class (neuron-direction pair)
lambda = zeros(n,r);
for i = 1:r
    lambda(:,i) = mean(train(:,:,i),1);
end

%% predict the reach direction of the test data
[m,n,r] = size(test);
clear g

for letter = 1:r
    for j = 1:m
        sample = test(j,:,letter);
        for i = 1:n
            for target = 1:r
                idx = (j-1)*r+target;
                g(i,idx,letter) = poisspdf(sample(i),lambda(i,target));
            end
        end
    end
end

%% sum log of probabilities 

log_g = log(g);
clear sum_log_g
for i = 1:r
    sum_log_g(:,i) = sum(log_g(:,:,i));
end

% assign samples to letter classes
clear test_class
for letter = 1:r
    for i = 1:m
        [~,test_class(i,letter)] = max(sum_log_g((i-1)*r+(1:r),letter));
    end
end

% calculate accuracy
correct = 0;
for target = 1:r
    correct = correct + sum(test_class(:,target) == target);
end
accuracy = correct / (m*r); 

%% Training and testing dataset

X_train = [];   X_test = [];
Y_train = [];   Y_test = [];

% start_idx = 1;
% for letter = 1:num_l
%     Y_train = [Y_train; data1(start_idx:start_idx + 2599,:)];
%     Y_test = [Y_test; data1(start_idx + 2600:start_idx + 2699,:)];
%     X_train = [X_train; velocity1(start_idx:start_idx + 2599,:)];
%     X_test = [X_test; velocity1(start_idx + 2600:start_idx + 2699,:)];
%     start_idx = start_idx + 2700;
% end

Y_train = data1(501:end, :);
Y_test = data1(1:500, :);
X_train = velocity1(501:end, :);
X_test = velocity1(1:500, :);

Y_train = im2double(Y_train);
Y_test = im2double(Y_test);

%% Linear decoder 

B_lr = inv(Y_train'*Y_train) * Y_train' * X_train;

X_pred = Y_test * B_lr;
MSE = mean((X_test - X_pred).^2);
% for i = 1:4
%     temp = corrcoef(X_test((:,i), X_pred(:,i));
%     corr(i) = temp(end,1);
% end

figure();
plot(X_test(1:100,1), X_test(1:100,2))
hold on; plot(X_pred(1:100,1), X_pred(1:100,2))

figure();
plot(X_test(101:200,1), X_test(101:200,2))
hold on; plot(X_pred(101:200,1), X_pred(101:200,2))

figure();
plot(X_test(201:300,1), X_test(201:300,2))
hold on; plot(X_pred(201:300,1), X_pred(201:300,2))

figure();
plot(X_test(301:400,1), X_test(301:400,2))
hold on; plot(X_pred(301:400,1), X_pred(301:400,2))

%%
figure();
plot(cumsum(X_test(1:100,1)), cumsum(X_test(1:100,2)))
hold on; plot(cumsum(X_pred(1:100,1)), cumsum(X_pred(1:100,2)))

figure();
plot(cumsum(X_test(101:200,1)), cumsum(X_test(101:200,2)))
hold on; plot(cumsum(X_pred(101:200,1)), cumsum(X_pred(101:200,2)))

figure();
plot(cumsum(X_test(201:300,1)), cumsum(X_test(201:300,2)))
hold on; plot(cumsum(X_pred(201:300,1)), cumsum(X_pred(201:300,2)))

figure();
plot(cumsum(X_test(301:400,1)), cumsum(X_test(301:400,2)))
hold on; plot(cumsum(X_pred(301:400,1)), cumsum(X_pred(301:400,2)))