#!/usr/bin/env python
# coding: utf-8



class XAI_lime():
    
    def __init__(self, model, model_name, preprocess_func, img_dim = 384):
        self.model = model
        self.model_name = model_name
        self.img_dim = img_dim
        self.preprocess_func = preprocess_func
        
    
    def _predict_fn(self, images):
        #function for input of explainer.explain_instance
        
        return self.model.predict(images, batch_size=16)
    
    def make_lime_heatmap(self, img):
        #Get explanation from lime
        
        img_preprocessed = self.preprocess_func(img)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_preprocessed.astype('double'), self._predict_fn, hide_color=0, num_samples=1000)
        _, heatmap = explanation.get_image_and_mask(7, positive_only = True)
        #label_index = 7 for polyp class
        
        
        return heatmap
    
    
    def lime_single_auroc(self, X_tst, Y_tst):
        #Calculate auroc for each image and mask
        
        heatmap = self.make_lime_heatmap(X_tst)
        fpr, tpr, _ = roc_curve(Y_tst.ravel(), heatmap.ravel())
        roc_auc = auc(fpr, tpr)
        
        return heatmap, roc_auc
    
    def lime_total_auroc(self, X_tst, Y_tst):
        #Calculate auroc for whole dataset
        heatmap_list = []
        auroc_list = []
        
        for i in range(len(X_tst)):
            heatmap, roc_auc = self.lime_single_auroc(X_tst[i], Y_tst[i])
            
            heatmap_list.append(heatmap)
            auroc_list.append(roc_auc)
        
        with open('./kvasir_lime_heatmaps_'+str(self.model_name)+'.pickle', 'wb') as f:
            pickle.dump([heatmap_list], f)
        
        self.total_auroc_completed = True
        
        
        return sum(auroc_list) / len(auroc_list)
    
    
    def lime_single_visualize(self, X_tst, Y_tst, alpha):
        img = X_tst.copy()
        heatmap = self.make_lime_heatmap(img)
        heatmap = cv2.resize(heatmap, (384, 384))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        
        
        
        superimposed_img = (heatmap * alpha + img * (1 - alpha)).astype('uint8')
        
        fig, axs = plt.subplots(1, 4, figsize=(3*12, 1*12))
        axs[0].imshow(heatmap)
        axs[0].set_title('Lime heatmap', fontsize = 30)
        axs[1].imshow(img)
        axs[1].set_title('Input Image', fontsize = 30)
        axs[2].imshow(superimposed_img)
        axs[2].set_title('Superimposed Image', fontsize = 30)
        axs[3].imshow(Y_tst)
        axs[3].set_title('Mask', fontsize = 30)

        for ax in axs.flat:
            ax.label_outer()
 
    
    
    def lime_single_visualize_faster(self, X_tst, Y_tst, alpha, idx):
        
        try:
            with open('./kvasir_lime_heatmaps_'+str(self.model_name)+'.pickle', 'rb') as f:
                total_heatmaps = pickle.load(f)
            
            img = X_tst.copy()    
            heatmap = total_heatmaps[0][idx]
            heatmap = cv2.resize(heatmap, (384, 384))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            superimposed_img = (heatmap * alpha + img * (1 - alpha)).astype('uint8')

            fig, axs = plt.subplots(1, 4, figsize=(3*12, 1*12))
            axs[0].imshow(heatmap)
            axs[0].set_title('Lime heatmap', fontsize = 30)
            axs[1].imshow(img)
            axs[1].set_title('Input Image', fontsize = 30)
            axs[2].imshow(superimposed_img)
            axs[2].set_title('Superimposed Image', fontsize = 30)
            axs[3].imshow(Y_tst)
            axs[3].set_title('Mask', fontsize = 30)

            for ax in axs.flat:
                ax.label_outer()
            
        
        
        except:
            raise KeyError('lime_total_auroc method should be done!')

