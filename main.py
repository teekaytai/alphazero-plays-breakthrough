from trainer import Trainer

MAX_EPOCHS = 100
lr=0.001
weight_decay=1e-4

def main():
    trainer = Trainer()
    trainer.train(MAX_EPOCHS, lr, weight_decay)

if __name__ == '__main__':
    main()
