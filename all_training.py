from models import simpleFCN


class allTraining():
    def __init__(self, sub_image_size, trainR0, trainL0, trainR1, num_patch=None):
        # getting the sub image size
        # initializing the model
        self.sub_image_size = sub_image_size
        # self.model = simpleFCN(sub_image_size *sub_image_size*3 *2)
        self.trainR0 = trainR0
        self.trainL0 = trainL0
        self.trainR1 = trainR1
        self.image_max_dim = next(iter(self.trainL0))[0].shape[0]
        self.num_patch = int(self.image_max_dim / self.sub_image_size)**2
        print(f"Max. dimension of Image : {self.image_max_dim}")
        print(f"\nMax. no. of Models/Patches possible are {self.num_patch}")
        if num_patch != None:
            self.num_patch = num_patch
        print("\nNo. of Models/patches selected : ", self.num_patch)

    def model_init(self, first_dim, last_dim):
        self.model_list = {}
        for i in range(self.num_patch):
            self.model_list['model_'+str(i+1)] = simpleFCN(first_dim, last_dim)
        print(f"\n{len(self.model_list)} Models initialized")
        return self.model_list

    def whole_training(self, model_list, patch_size_list, criterion, epochs, training_model):
        self.model_list = model_list
        data_rec = {}
        patch_list = {}
        print(f"\n{self.num_patch} no. of models will be trained. ")
        for p in range(self.num_patch):

            model = self.model_list['model_'+str(p+1)]
            patch = patch_size_list['Patch_'+str(p+1)]
            patch_list['Patch_'+str(p+1)] = patch
            print(
                f"\nModel No. {p+1} | Patch No. {p+1} | Patch Size : {patch}")
            print(f"Training started .... ")
            # print(model)
            # print(patch_list)
            epoch_list, loss_list = training_model(self.trainR0, self.trainL0, self.trainR1, model, patch, criterion, epochs,
                                                   opt=optim.Adam(model.parameters(), lr=0.00001))

            data_rec['M'+str(p+1)] = list((epoch_list, loss_list))
            print(f"Training no. ended .... \n\n\n")

        return self.model_list, patch_list, data_rec
