from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from sklearn.model_selection import KFold
import copy
from utils import *
from models import *
from dataloader import *


parser = argparse.ArgumentParser(description='Persuasiveness')
parser.add_argument('--data-dir', default='./data', help='path to data')
parser.add_argument('--exp-dir', default='./experiments/debug', help='path save experimental results')
parser.add_argument('--exp-mode', default=0, choices=[0,1,2,3,4,5,6], type=int, help='0:persuasive; 1:image content; 2: stance; 3: perusasion_mode; 4:persuasive mode logos; 5:persuasive mode panthos; 6:persuasive mode ethos')
parser.add_argument('--data-mode', default=2, choices=[0,1,2], type=int, help='0:text; 1:image; 2:image+text')
parser.add_argument('--gpus', default='1', type=str, help='specified gpus')
parser.add_argument('--batch-size', default=16, type=int, help='number of samples per batch')
parser.add_argument('--persuasive-label-threshold', default=0.6, type=float, help='threshold to categorize persuasive labels')
parser.add_argument('--kfold', default=5, help='number of fold validation')
parser.add_argument('--img-model', default=0, choices=[0,1,2], type=int, help='0:Resnet50; 1:Resnet101; 2:VGG16')
parser.add_argument('--seed', default=22, type=int, help='random seed number')
parser.add_argument('--text-id', default=1330962451215638528, type=int, help='the id of sample to show')
parser.add_argument('--top-k', default=5, type=int, help='top images to show')
args = parser.parse_args()


# parameters
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"
exp_name = get_exp_name(args)
args.exp_dir = f"./experiments/{exp_name}"
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

df_raw = pd.read_csv(os.path.join(args.data_dir, 'gun_control_annotation.csv'), index_col=0)
text = df_raw[df_raw["tweet_id"] == args.text_id]["tweet_text"].tolist()[0]
df_exp = df_raw.copy()
df_exp["tweet_text"] = text


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

if args.exp_mode == 1:
    # multi-class classification
    if args.img_model == 0:
        init_model = MultiModelResnet50(out_dim=6)
    elif args.img_model == 1:
        init_model = MultiModelResnet101(out_dim=6)
    else:
        init_model = MultiModelVGG16(out_dim=6)
    criterion = nn.CrossEntropyLoss()
elif args.exp_mode == 3:
    # multi-label classification
    if args.img_model == 0:
        init_model = MultiModelResnet50(out_dim=3)
    elif args.img_model == 1:
        init_model = MultiModelResnet101(out_dim=3)
    else:
        init_model = MultiModelVGG16(out_dim=3)
    criterion = nn.BCEWithLogitsLoss()
else:
    # binary classification
    if args.img_model == 0:
        init_model = MultiModelResnet50(out_dim=1)
    elif args.img_model == 1:
        init_model = MultiModelResnet101(out_dim=1)
    else:
        init_model = MultiModelVGG16(out_dim=1)
    criterion = nn.BCEWithLogitsLoss()


output_df = []
kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
for fold, (train_idx, val_idx) in enumerate(kfold.split(df_exp)):
    print('Running fold {}...'.format(fold + 1))
    if not os.path.exists(os.path.join(args.exp_dir, f"fold_{fold}_model_best.pth.tar")):
        print("skip...")
        continue

    checkpoint = torch.load(os.path.join(args.exp_dir, f"fold_{fold}_model_best.pth.tar"))
    init_model.load_state_dict(checkpoint['state_dict'])
    model = copy.deepcopy(init_model)
    model.to(device)
    model.eval()

    val_annotation = df_exp.iloc[val_idx].reset_index()
    val_dataset = ImageTextDataset(args, annotation=val_annotation, root_dir=os.path.join(args.data_dir, 'images'), transform=val_transform)
    val_dataloaders = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    predicted_labels = []
    predicted_probs = []
    predicted_image_ids = []
    gold_labels = []

    with torch.no_grad():
        for i, (img_ids, input_ids, attention_masks, image, labels) in enumerate(val_dataloaders):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            image = image.to(device)

            logits = model(input_ids, attention_masks, image)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            preds = outputs.reshape(-1).round()

            predicted_image_ids += list(img_ids)
            predicted_labels += preds.detach().cpu().tolist()
            predicted_probs += outputs.reshape(-1).detach().cpu().tolist()
            gold_labels += labels.reshape(-1).detach().cpu().tolist()

    predict_df = pd.DataFrame({"image_ids":predicted_image_ids, "predicted_labels":predicted_labels, "probabilities": predicted_probs})
    predict_df["text_ids"] = args.text_id
    predict_df["text"] = text
    print(predict_df)
    output_df.append(predict_df)

output_df = pd.concat(output_df, axis=0)
output_df = output_df.sort_values(by=['probabilities'], ascending=False)
output_df.to_csv(os.path.join(args.exp_dir, f"demo_{args.text_id}.csv"))
print(output_df)

rank_ids = output_df["image_ids"].tolist()
rank_prob = output_df["probabilities"].tolist()
cand_ids = rank_ids[:args.top_k]
cand_probs = rank_prob[:args.top_k]
plot_image(args, cand_ids, cand_probs, option=exp_name)
