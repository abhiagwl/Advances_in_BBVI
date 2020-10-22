def record_if_good(model_path,i):
    import mystan
    import pickle
    rez = mystan.is_good_model(model_path)
    file = open('good-models/' + str(i) + '.pkl','wb')
    pickle.dump((model_path,rez),file)
    file.close()


if __name__=='__main__':
    import mystan
    import ezrun
    import os
    import glob
    import pickle

    if not os.path.exists('good-models'):
        print('creating good-models/ to store what models are good')
        os.makedirs('good-models')

    model_paths = mystan.all_model_paths()

    for i,model_path in enumerate(model_paths):
        #record_if_good(model_path,i)
       ezrun.run(record_if_good,model_path,i,mem=10000,ncores=1,hours=0,minutes=5)

    ezrun.wait_for_complete()

    good_model_paths = []

    for i,model_path in enumerate(model_paths):
        try:
            file = open('good-models/' + str(i) + '.pkl','rb')
            stuff,rez = pickle.load(file)
            file.close()
            print('stuff',stuff,'rez',rez)
            if rez:
                good_model_paths.append(model_path)
        except FileNotFoundError as e:
            print('file not found',e)

    print('here are all the good models',good_model_paths)
    print(good_model_paths)
    print('num good models',len(good_model_paths))
    file = open('good_model_paths.pkl','wb')
    pickle.dump(good_model_paths,file)
    file.close()
