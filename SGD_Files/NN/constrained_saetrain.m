function sae = constrained_saetrain(sae, x, opts, x_without_noise,val_x,val_y)

    for i = 1 : numel(sae.ae);
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
        [sae.ae{i},sae.ae{i}.Loss] = constrained_nntrain(sae.ae{i}, x, x_without_noise, opts,val_x,val_y);
        t = nnff(sae.ae{i}, x, x_without_noise);
        x = t.a{2};
        %remove bias term
        x = x(:,2:end);
    end
end
