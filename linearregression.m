clear;
veldata = load('computerMouseTemplates.mat');
corticaldata = load('t5.2019.05.08/singleLetters.mat');
%%
train_count = 5;
Y_tr = concatCorticalSignals2(train_count);
X_tr = concatInterpVels2(train_count);
% Y_te = concatCorticalSignals(26);
Y_te = corticaldata.neuralActivityCube_c(1,51:151,:);
Y_te = squeeze(Y_te);
%% linear regression training
% B = inv(Y_tr'*Y_tr)*Y_tr'*X_tr;
B = Y_tr\X_tr;
X_te = im2double(Y_te)*im2double(B);
%% linear regression testing
figure(1);
plot(1000*X_te(:,1),'-r');
hold on;
plot(veldata.b(:,1),'-b');

%%
function [interped] = interpmatrix(vel)
    x0 = vel(:,1);
    x0_ns = linspace(1,length(x0),length(x0));
    x1_ns = linspace(1,100,100);
    x1 = (interp1(x0_ns, x0, x1_ns, 'linear','extrap'))';
%     FX = griddedInterpolant(x0_ns, x0);
%     x1 = FX(x1_ns)';
    
    y0 = vel(:,2);
    y0_ns = linspace(1,length(y0),length(y0));
    y1_ns = linspace(1,100,100);
    y1 = (interp1(y0_ns, y0, y1_ns, 'linear','extrap'))';
%     FY = griddedInterpolant(y0_ns, y0);
%     y1 = FY(y1_ns)';
    
    interped = [x1 y1];
end

function [veldata] = concatInterpVels()
    data = load('computerMouseTemplates.mat');
    veldata = zeros([100*32,2]);
    letters = fieldnames(data);
    for i=1:32
        if size(data.(letters{i}),2) == 2            
            interpvels = interpmatrix(data.(letters{i}));
            veldata(((i-1)*100)+1:(i*100),:) = interpvels;
        end       
    end
end

function [corticalsignals] = concatCorticalSignals(trial)
    data = load('t5.2019.05.08/singleLetters.mat');
    ts_count = 100;
    ts_gocue = 51;
    letters = fieldnames(data);
    counter = 1;
    corticalsignals = zeros(32*100,192);
    for i = 1:length(letters)
        if startsWith(letters{i}, 'neuralActivityCube')
            letterdata = data.(letters{i});    
            letterdata = letterdata(:,ts_gocue:ts_gocue+ts_count-1,:);
            corticalsignals(((counter-1)*100)+1:(counter*100),:) = letterdata(trial,:,:);
            counter = counter+1;
        end
    end

end

function [corticalsignals2] = concatCorticalSignals2(train_count)
    corticalsignals2 = zeros([train_count*32*100,192]);
    for i = 1:train_count
        corticalsignals2(((i-1)*32*100)+1:i*32*100,:) = concatCorticalSignals(i);
    end
end

function [veldata2] = concatInterpVels2(train_count)
    veldata2 = zeros([train_count*32*100,2]);
    for i = 1:train_count
        veldata2(((i-1)*32*100)+1:i*32*100,:) = concatInterpVels();
    end
end

function [pos] = vel2pos(vel)
    letter = vel;
    pos = zeros(size(letter));
    ts = 100/1000;
    for i=2:length(letter)
        xpos = letter(i,1)*ts + pos(i-1,1);
        ypos = letter(i,2)*ts + pos(i-1,2);
        pos(i,1) = xpos;
        pos(i,2) = ypos;
    end
end

function [] = plotPositions(vel)
    letter = vel;
    pos = zeros(size(letter));
    ts = 100/1000;
    for i=2:length(letter)
        xpos = letter(i,1)*ts + pos(i-1,1);
        ypos = letter(i,2)*ts + pos(i-1,2);
        pos(i,1) = xpos;
        pos(i,2) = ypos;
    end
    scatter(pos(:,1),pos(:,2));
end

%% extra stuff
% vel = veldata.b;
% pos = vel2pos(vel)
% ip = interpmatrix(pos);
% scatter(ip(:,1),ip(:,2));

