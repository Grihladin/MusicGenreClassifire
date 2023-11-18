import json
import os
import math
import librosa

DATASET_PATH = "путь к датасету"
JSON_PATH = "путь к json файлу для записи"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # время в секундах
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Извлекает MFCC из набора музыкальных данных и сохраняет их в файл JSON вместе с метками жанров.

         :param dataset_path (str): путь к набору данных.
         :param json_path (str): путь к файлу json, используемому для сохранения MFCC.
         :param num_mfcc (int): количество коэффициентов для извлечения.
         :param n_fft (int): Интервал, для применения FFT. Измеряется в количестве образцов
         :param hop_length (int): скользящее окно для FFT. Измеряется в количестве образцов
         :param: num_segments (int): количество сегментов, на которые разделятся треки сэмплов.
        """

    # словарь для хранения mapping, labels, и MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # просмотреть все подпапки жанра
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # проверка, что мы обрабатываем уровень подпапки жанра
        if dirpath is not dataset_path:

            # сохранить метку жанра (т. е. имя подпапки) в mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # обработать все аудиофайлы в подкаталоге жанра
            for f in filenames:

		# загрузить аудиофайл
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # обработать все сегменты аудиофайла
                for d in range(num_segments):

                    # рассчитать начальную и финишную выборку для текущего сегмента
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # извлечь mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], 
                                                sr=sample_rate, 
                                                n_fft=n_fft, 
                                                n_mfcc=num_mfcc, 
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # сохранять только mfcc с ожидаемым количеством векторов
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

    # сохранить MFCC в файл JSON
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)