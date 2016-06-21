function patchpix = NLextract(I,searchWindow,patchWindow)
zer = (patchWindow - 1)/2;                          %padding the image to ensure allpixels are 
                                                    %covered by search window
I1 = padarray(I,[zer zer],'replicate');
[m,n] = size(I);
count2 = 1;
for i = 1:m
    for j = 1:n
        i1 = i + zer;
        j1 = j + zer;
        patch = I1(i1-zer:i1+zer,j1-zer:j1+zer);
        rmin = max(i1-searchWindow,zer+1);                   %setting the search window around the given patch 
        rmax = min(i1+searchWindow,m+zer);
        smin = max(j1-searchWindow,zer+1);
        smax = min(j1+searchWindow,n+zer);
        count = 1;
        for r=rmin:1:rmax
            for s=smin:1:smax
                W = I1(r-zer:r+zer,s-zer:s+zer);
                d(1,count) = sum(sum((patch - W).*(patch - W))); %this needs to be a row vector
                feat(:,count) = reshape(W,[9,1]);                %matrix of features, column size count, row size 9
                count = count +1;
            end
        end
        intmat = [d;feat];         % number of features generated is equal to the number of distances generated
        intmat = intmat';          % transpose to use sortrows
        intmat = sortrows(intmat); % sort on the basis of the distance
        patchpix(:,:,count2) = intmat(1:49,2:10); % third dimension indicates each pixel
                                                  % each row in each matrix is the nearest patch.  
        count2 = count2 + 1;
    end
end

        
        
                
                
        
