function sae = saetrain(sae, x, opts, x_without_noise)

    for i = 1 : numel(sae.ae);
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
        [sae.ae{i},sae.ae{i}.Loss] = nntrain(sae.ae{i}, x, x_without_noise, opts);
        t = nnff(sae.ae{i}, x, x_without_noise);
        x = t.a{2};
        %remove bias term
        x = x(:,2:end);
    end
end
