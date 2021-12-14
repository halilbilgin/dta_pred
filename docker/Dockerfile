FROM continuumio/anaconda3

RUN mkdir model #make directory for scripts
RUN mkdir input #make input directory
RUN mkdir output #make output directory
#RUN mkdir bin #make directory for scripts

RUN /bin/bash -c "cd /model && \
    git clone https://github.com/halilbilgin/dta_pred.git && \
    cd dta_pred && \
    git checkout dev && \
    conda install keras tensorflow matplotlib numpy && \
    python setup.py develop && \
    cp data/leaderboards/round2/template.csv /input/input.csv"
	
ENTRYPOINT ["python", "/model/dta_pred/bin/predict_pkd.py", "--model_file=/model/dta_pred/trained_models/dta_pred_v2/Kd.h5", "--input_file=/input/input.csv", "--output_file=/output/predictions.csv", "--arguments={'apply_bn':0,'batch_size':128,'binary_th':7,'biovec_model_path':'trained_models/biovec/swissprot-reviewed.model','biovec_output_dim':300,'checkpoints_path':'output/experiment_13.12.21-16.31.13/checkpoints','cross_validation':false,'dataset_path':'data/','datasets_included':['dtc_KD','kiba','dtc_IC50','davis'],'dropout':0.5,'drug_format':'labeled_smiles','experiment_name':'experiment','l2_regularizer_fc':0,'learning_rate':0.001,'log_path':'output/experiment_04.18.19-16.31.13/logs','loss':'crossentropy_mse_combined','max_seq_len':1500,'max_smi_len':120,'mol2vec_model_path':'trained_models/mol2vec/model_300dim.pkl','mol2vec_output_dim':300,'mol2vec_radius':1,'mongodb':'217.112.83.250:80:DTA_PRED','n_cnn_layers':3,'n_fc_layers':1,'n_fc_neurons':1024,'num_epoch':300,'num_windows':96,'optimizer':'adam','output_path':'output/experiment_04.18.19-16.31.13','protein_format':'sequence','seed':42,'seq_model':'inception','seq_window_length':4,'smi_model':'inception','smi_window_length':4}"]
