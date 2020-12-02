def testDmdVlad(opts, DMDopts, FVopts):


    numTrain = 5000 ;
    train = vl_colsubset(find(imdb.images.set <= 2), numTrain, 'uniform') ;
    paths = cellfun(@(S) fullfile(imdb.imageDir, S), imdb.images.name(train), ...
        'Uniform', 0);

    centers = trainEncoder(paths, FVopts, DMDopts) ;

    descrs = encodeVlad(centers, cellfun(@(S) fullfile(imdb.imageDir, S), ...
        imdb.images.name, 'Uniform', 0), DMDopts, FVopts) ;

    if isfield(imdb.images, 'class')
      classRange = unique(imdb.images.class) ;
    else
      classRange = 1:numel(imdb.classes.imageIds) ;
    end
    numClasses = numel(classRange) ;


    descrs = bsxfun(@times, descrs, 1./sqrt(sum(descrs.^2))) ;

    % train and test
    % 1 - training data; 2 - validation; 3 - test;
    % for training, we use train+val
    train = find(imdb.images.set <= 2) ;
    test = find(imdb.images.set == 3) ;

    lambda = 1 / (opts.C*numel(train)) ;
    par = {'Solver', 'sdca', 'Verbose', ...
           'BiasMultiplier', 1, ...
           'Epsilon', 0.001, ...
           'MaxNumIterations', 100 * numel(train)} ;

    scores = cell(1, numel(classRange)) ;
    ap = zeros(1, numel(classRange)) ;
    ap11 = zeros(1, numel(classRange)) ;
    w = cell(1, numel(classRange)) ;
    b = cell(1, numel(classRange)) ;
    for c = 1:numel(classRange)
      if isfield(imdb.images, 'class')
        y = 2 * (imdb.images.class == classRange(c)) - 1 ;
      else
        y = - ones(1, numel(imdb.images.id)) ;
        [~,loc] = ismember(imdb.classes.imageIds{classRange(c)}, imdb.images.id) ;
        y(loc) = 1 - imdb.classes.difficult{classRange(c)} ;
      end
      if all(y <= 0), continue ; end

      [w{c},b{c}] = vl_svmtrain(descrs(:,train), y(train), lambda, par{:}) ;
      scores{c} = w{c}' * descrs + b{c} ;

      [~,~,info] = vl_pr(y(test), scores{c}(test)) ;
      ap(c) = info.ap ;
      ap11(c) = info.ap_interp_11 ;
      fprintf('class %s AP %.2f; AP 11 %.2f\n', imdb.meta.classes{classRange(c)}, ...
              ap(c) * 100, ap11(c)*100) ;
    end
    scores = cat(1,scores{:}) ;
    % -------------------------------------------------------------------------


    diary off ;
    diary on ;

    % confusion matrix (can be computed only if each image has only one label)
    if isfield(imdb.images, 'class')
      [~,preds] = max(scores, [], 1) ;
      confusion = zeros(numClasses) ;
      for c = 1:numClasses
        sel = find(imdb.images.class == classRange(c) & imdb.images.set == 3) ;
        tmp = accumarray(preds(sel)', 1, [numClasses 1]) ;
        tmp = tmp / max(sum(tmp),1e-10) ;
        confusion(c,:) = tmp(:)' ;
      end
    else
      confusion = NaN ;
    end;

    save(fullfile(opts.resultDir, sprintf('result-.mat')), ...
         'scores', 'ap', 'ap11', 'confusion', 'classRange', 'opts') ;


    % figures
    meanAccuracy = sprintf('mean accuracy: %f\n', mean(diag(confusion)));
    mAP = sprintf('mAP: %.2f %%; mAP 11: %.2f', mean(ap) * 100, mean(ap11) * 100) ;

    if (1 == nargout)
      results.mAP = mean(ap);
      results.mAcc = mean(diag(confusion));
    end


    if 0
    figure(1) ; clf ;
    imagesc(confusion) ; axis square ;
    title([opts.prefix ' - ' meanAccuracy]) ;
    vl_printsize(1) ;
    print('-dpdf', fullfile(opts.resultDir, sprintf('result-confusion.pdf'))) ;
    print('-djpeg', fullfile(opts.resultDir, sprintf('result-confusion.jpg'))) ;
    figure(2) ; clf ; bar(ap * 100) ;
    title([opts.prefix ' - ' mAP]) ;
    ylabel('AP %%') ; xlabel('class') ;
    grid on ;
    vl_printsize(1) ;
    ylim([0 100]) ;
    print('-dpdf', fullfile(opts.resultDir, sprintf('result-ap.pdf'))) ;

    end
    disp(meanAccuracy) ;
    disp(mAP) ;
    diary off ;

    if isfield(imdb.images, 'class')
      [~,preds] = max(scores, [], 1) ;
      for cc = 1:numClasses
        sel = find(imdb.images.class == classRange(cc) & imdb.images.set == 3) ;
      end
    else:
      confusion = NaN ;
    end
