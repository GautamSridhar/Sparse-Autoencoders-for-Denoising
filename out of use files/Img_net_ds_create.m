number_of_images = 10;
j=1;
for i = 1:number_of_images
    if(i<10)
        str2 = 'ILSVRC2012_test_0000000';
    elseif(i>=10 && i<100)
        str2 = 'ILSVRC2012_test_000000';
    else
        str2 = 'ILSVRC2012_test_00000';
    end
    str4 = num2str(i);
    str3 = '.JPEG';
    str1 = strcat(str2, str4, str3); %forming the input images name
    img = imread(str1);
    if(size(img,3)==3 )
        img =rgb2gray(img);
    end
    if (size(img,1)>256 && size(img,2)>256)
        IMAGES_IN(:,:,j) = img(1:256,1:256);
        j=j+1;
    end
end

save IMAGES_IN ;