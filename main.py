from trainer import Trainer

MAX_EPOCHS = 100

def main():
    trainer = Trainer()
    trainer.train(MAX_EPOCHS)

if __name__ == '__main__':
    main()
