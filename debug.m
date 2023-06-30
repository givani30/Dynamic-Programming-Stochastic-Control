nx=2; %number of states: first state is the budget, second state is the team value

N=500; %number of players
%%

fileName = 'project_4_data.json'; % filename in JSON extension
fid = fopen(fileName); % Opening the file
raw = fread(fid,inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid); % Closing the file
data = jsondecode(str); % Using the jsondecode function to parse JSON from string
T=struct2table(data)
T=convertvars(T,@iscellstr,"string")
T.position=categorical(T.position);

%% 
% 

%Create S matrix

pos_idx=grp2idx(T.position);
positions=categories(T.position)
S=zeros(length(positions),N);
for i=1:height(T)
    S(pos_idx(i),i)=1;
end

%% 
% Extend p vectors

T_new=T;
%Convert date to year
years=[2004:2021];
p_mat=NaN(N,length(years))
for i=1:height(T)
    historynew=struct2table(T.history{i});
    historynew.date=year(historynew.date);
    % Compute group summary
    historynew = groupsummary(historynew,"date","mean","value");
    historynew.GroupCount = [];

    historynew.date=categorical(historynew.date,years);
    year_idx=grp2idx(historynew.date);
    p_bar_i=NaN(1,length(years));
    for j=1:length(year_idx)
        p_bar_i(year_idx(j))=historynew.mean_value(j);
    end
    p_bar_i=fillmissing(p_bar_i,"nearest");
    p_mat(i,:)=p_bar_i;
end
p_mat

p_bar=p_mat(:,8:end-1) %select from 2011 to 2020
%% 
% Z matrix

Z=NaN(N,9)
for i=1:9
    Z(:,i)=(p_bar(:,i+1)-p_bar(:,i))./p_bar(:,i);
end
Z=fillmissing(Z,'constant',0)
Z(isinf(Z))=0
%%
yalmip("clear")
T_end=10; %number of years

%% 
% Model data

x=sdpvar(repmat(nx,1,T_end),repmat(1,1,T_end));
u=binvar(repmat(N,1,T_end),repmat(1,1,T_end));
v=sdpvar(repmat(N,1,T_end),repmat(1,1,T_end));
%% 
% Formulate optimization problem

constraints=[]
J{T_end}=x{T_end}(1)
objective=0;
ops = sdpsettings('verbose',3,'debug','on','removeequalities',1)
%Terminal constraint 
constraints=[constraints,u{T_end}==0,
            x{T_end}(2)==sum(v{T_end})]

% for k=T_end-1:-1:1
for k=9
    %Feasible region
    constraints=[constraints,S*u{k}==2,
        x{k}(2)==sum(v{k}),
        0<=x{k}<=1e10];

    %Dynamics
    constraints=[constraints,x{k+1}(1)==x{k}(1)+(u{k}-u{k+1})'*p_bar(:,k),
        v{k+1}==v{k}+v{k}.*Z(:,k)];
    objective=objective-J{k+1};
    [sol{k},dgn{k},Uz{k},J{k},uopt{k}] = solvemp(constraints,[],ops,x{k},u{k})
end