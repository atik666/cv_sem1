function Per=accuracy(Truth,Img_out)

[X, Y]=size(Truth);

Z=zeros(256,256);

for i=1:X
    for j=1:Y
        p=Truth(i,j)+1;
        q=ceil(Img_out(i,j));
        Z(p,q)=Z(p,q)+1;
    end
end

T=sum(max(Z));

Per=T/X/Y;