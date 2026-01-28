#import "raport.typ": *

#set heading(numbering: "1.")

#set table(
  fill: (_, y) => if calc.odd(y) { rgb("EAF2F5") },
  stroke: 0.3pt + black.lighten(10%),
)

#show: conf.with(
  subject: [Przetwarzanie danych z wykorzystaniem GPU],
  title: [Projekt Detekcja Mowy],
  subtitle: [Styczeń 2026],
  lang: "pl",
  pagenum: "1.",
)

#v(5em)

= Wstęp

Celem projektu były przygotowanie aplikacji,
która na podstawie wejściowego sygnału audio wykrywać predefiniowane słowa.

Podstawowymi elementami projektu było:
- Przygotowanie modelu z wykorzystaniem biblioteki `PyTorch` pozwalającego na wykrywanie mowy
- Skwantyzowanie modelu i porównanie wyników
- Przygotowanie aplikacji w C++, która wykorzystuje przygotowany model za pomocą biblioteki `ExecuTorch`

Projekt dostępny jest na platformie GitHub pod adresem #box[https://github.com/rosowskimik/subcommands-put]

= Zbiór Danych

Do treningu modelu został wykorzystany zbiór `Speech Commands` (https://arxiv.org/abs/1804.03209) w wersji `v0.02`. Zbiór ten zawiera pliki audio w formacie `.wav`.
Pliki te są podzielone na zbiór konkertnych komend.
Każdy plik zawiera 1 sekundę nagrania wymowy jednego wyrazu w języku angielskim lub szum tła.
Słowa są wymawiane przez różne osoby.
Dodatkowo do każdego nagrania przypisany jest numer identyfikacyjny mówiącego `speaker_id`,
indeks wymowy `utterance_id` na każdego mówiącego, etykieta z wymawianym słowem `label`,
wartość logiczna czy wypowiadane słowo jest uznawane za prostą komendę `is_unknown` oraz częstotliwość próbkowania pliku audio `sampling_rate` (@sample_data).

#figure(
  code(
    ```json
    {
      "file": "no/7846fd85_nohash_0.wav",
      "audio": {
        "path": "no/7846fd85_nohash_0.wav",
        "array": array([ -0.00021362, -0.00027466, -0.00036621, ...,  0.00079346,
              0.00091553,  0.00079346]),
        "sampling_rate": 16000
        },
      "label": 1,  # "no"
      "is_unknown": False,
      "speaker_id": "7846fd85",
      "utterance_id": 0
    }
    ```,
  ),
  caption: [Przykład danych dla jednej z wymów słowa _"no"_],
  placement: auto,
) <sample_data>

#pagebreak()

Cały zbiór danych (w wersji `0.02`) zawiera $105 829$ plików audio,
podzielonych na 36 etykiet:

- "Backward"
- "Bed"
- "Bird"
- "Cat"
- "Dog"
- "Down"
- "Eight"
- "Five"
- "Follow"
- "Forward"
- "Four"
- "Go"
- "Happy"
- "House"
- "Learn"
- "Left"
- "Marvin"
- "Nine"
- "No"
- "Off"
- "On"
- "One"
- "Right"
- "Seven"
- "Sheila"
- "Six"
- "Stop"
- "Three"
- "Tree"
- "Two"
- "Up"
- "Visual"
- "Wow"
- "Yes"
- "Zero"

Dodatkowo istnieje także specjalna etykieta `_silence_`.
Nagrania oznaczone tą etykietą są albo nagraniami albo matematyczną symulacją szumu.

Zbiór `Speech Commands v0.02` jest dostarczany z podziałem na trzy rozłączne
podzbiory: treningowy (`training`), walidacyjny (`validation`) oraz testowy
(`testing`). W projekcie wykorzystano dokładnie ten podział udostępniany przez
dataset, bez ręcznego mieszania przykładów pomiędzy subsetami. Liczność
przykładów w poszczególnych częściach wynosiła odpowiednio:

- `train`: 84848
- `validation`: 9982
- `testing`: 4890

= Transformacja danych

Zbiór `Speech Commands` zawiera surowe nagrania w formacie `.wav`, których nie da
się bezpośrednio podać na wejście klasycznego modelu konwolucyjnego. Z tego
powodu każdy przykład został przekształcony do postaci dwuwymiarowej
reprezentacji czas–częstotliwość (spektrogram Mel), o stałym rozmiarze (@mel_spec).

Dane zostały wczytane nie bezpośrednio z surowego archiwum pobranego ze strony
źródłowej, lecz z wykorzystaniem klasy `torchaudio.datasets.SPEECHCOMMANDS`,
czyli przygotowanego pod PyTorcha wrappera na ten sam zbiór danych. Zapewnia on
automatyczne pobranie i rozpakowanie zbioru oraz zwraca próbki w postaci tensora
`waveform` wraz z metadanymi (m.in. `sample_rate`, `label`, `speaker_id`,
`utterance_number`), co upraszcza dalsze przetwarzanie i trening.

Dla każdego pliku audio wykonywane były następujące kroki:

- *Resampling:* jeżeli częstotliwość próbkowania różniła się od docelowej,
  sygnał był przeliczany do `16 kHz`.
- *Ujednolicenie długości:* nagrania były przycinane lub dopełniane zerami do
  dokładnie 1 sekundy (`16000` próbek). Zapewnia to stały rozmiar wejścia
  niezależnie od konkretnego pliku.
- *Wyznaczenie spektrogramu Mel:* z tak przygotowanego sygnału wyliczany był
  spektrogram Mel z parametrami: `n_mels = 40`, `n_fft = 512`,
  `hop_length = 256` (co dla 1 sekundy daje `63` ramek czasowych).
- *Skala logarytmiczna:* wartości mocy zostały przekształcone do skali
  decybelowej (logarytmicznej) za pomocą `AmplitudeToDB`, co poprawia
  interpretowalność i zwykle ułatwia uczenie (duże różnice amplitud są
  "spłaszczane").
- *Etykietowanie:* etykieta tekstowa słowa była mapowana na indeks klasy
  (liczbę całkowitą) zgodnie z listą `labels`.

Aby przyspieszyć kolejne uruchomienia i uniknąć wielokrotnego
przeliczania tych samych transformacji, przetworzone dane były zapisywane do
pamięci podręcznej (cache) w postaci plików `.pt`. Każdy plik zawierał parę:
`mel_spec` o rozmiarze `(40, 63)` oraz `label` jako indeks klasy. Następnie
podczas treningu modelu wykorzystywany był własny `Dataset`, który wczytywał
gotowe spektrogramy z cache, a `DataLoader` łączył je w batch o rozmiarze
`(B, 40, 63)`.

Dodatkowo ze zbioru treningowego wydzielona została niewielka część danych
kalibracyjnych (ok. 10%), która została wykorzystana na etapie kalibracji
podczas kwantyzacji post-training (PTQ).

#figure(
  image("img/mel_spec.png"),
  caption: [Wizualizacja spektogramu Mel po transformacji],
) <mel_spec>

= Architektura modelu

W projekcie wykorzystano niewielką sieć konwolucyjną typu CNN przeznaczoną do
klasyfikacji słów na podstawie reprezentacji czas–częstotliwość (spektrogramu
Mel). Architektura była inspirowana modelem do klasyfikacji audio prezentowanymi w dokumentacji MathWorks
(Deep Learning, przykłady dla rozpoznawania komend głosowych).

Wejściem modelu jest macierz spektrogramu o rozmiarze `(40, 63)` (liczba banków
Mel × liczba ramek czasowych). Ponieważ warstwy konwolucyjne 2D oczekują kanału,
dane są traktowane jako obraz jednokanałowy (kanał o rozmiarze 1).

Model składa się z pięciu bloków konwolucyjnych. Każdy blok ma postać:
`Conv2d(3×3) → BatchNorm2d → ReLU`, po którym wykonywane jest uśredniające
próbkowanie w dół `AvgPool2d(2)`. Liczba kanałów rośnie w pierwszych etapach
(1→12→24→32), a następnie pozostaje stała na poziomie 32. Taka konstrukcja
pozwala stopniowo redukować wymiar przestrzenny (czas i częstotliwość) aż do
reprezentacji `1×1`, którą następnie można spłaszczyć do wektora cech i podać na
warstwę klasyfikującą.

Ostatnią warstwą jest `Linear`, która mapuje 32 cechy na 37 klas (liczba etykiet
w zbiorze danych). Cały model jest bardzo lekki (ok. 29.5 tys. parametrów), co
ułatwia późniejszą kwantyzację oraz uruchamianie na urządzeniach o ograniczonych
zasobach.

#figure(
  code(
    ```txt
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 12, 40, 63]             108
           BatchNorm2d-2           [-1, 12, 40, 63]              24
                  ReLU-3           [-1, 12, 40, 63]               0
            ConvBnReLU-4           [-1, 12, 40, 63]               0
             AvgPool2d-5           [-1, 12, 20, 31]               0
                Conv2d-6           [-1, 24, 20, 31]           2,592
           BatchNorm2d-7           [-1, 24, 20, 31]              48
                  ReLU-8           [-1, 24, 20, 31]               0
            ConvBnReLU-9           [-1, 24, 20, 31]               0
            AvgPool2d-10           [-1, 24, 10, 15]               0
               Conv2d-11           [-1, 32, 10, 15]           6,912
          BatchNorm2d-12           [-1, 32, 10, 15]              64
                 ReLU-13           [-1, 32, 10, 15]               0
           ConvBnReLU-14           [-1, 32, 10, 15]               0
            AvgPool2d-15             [-1, 32, 5, 7]               0
               Conv2d-16             [-1, 32, 5, 7]           9,216
          BatchNorm2d-17             [-1, 32, 5, 7]              64
                 ReLU-18             [-1, 32, 5, 7]               0
           ConvBnReLU-19             [-1, 32, 5, 7]               0
            AvgPool2d-20             [-1, 32, 2, 3]               0
               Conv2d-21             [-1, 32, 2, 3]           9,216
          BatchNorm2d-22             [-1, 32, 2, 3]              64
                 ReLU-23             [-1, 32, 2, 3]               0
           ConvBnReLU-24             [-1, 32, 2, 3]               0
            AvgPool2d-25             [-1, 32, 1, 1]               0
               Linear-26                   [-1, 37]           1,221
    ================================================================
    Total params: 29,529
    Trainable params: 29,529
    Non-trainable params: 0
    ```,
  ),
  caption: [Struktura modelu],
) <model>

= Proces kwantyzacji

= Opis procesu kwantyzacji

Celem kwantyzacji było zmniejszenie rozmiaru modelu oraz przygotowanie go do
uruchomienia w aplikacji C++ z wykorzystaniem `ExecuTorch` i backendu `XNNPACK`.
W projekcie zastosowano kwantyzację typu post-training (PTQ).

Proces kwantyzacji składał się z kilku etapów:

- *Przygotowanie modelu do kwantyzacji (tryb inference):* model został
  przełączony w tryb `eval()`. Dodatkowo wykonano “folding” warstw
  `BatchNorm2d` do poprzedzających je `Conv2d` (fuzja `Conv+BN`), co upraszcza
  graf obliczeń i jest korzystne w kontekście kwantyzacji.

- *Eksport grafu:* model FP32 został wyeksportowany za pomocą `torch.export` do
  postaci modułu o stałym grafie (GraphModule). Wykorzystano przykładowe wejście
  o rozmiarze `(B, 40, 63)` oraz dopuszczono dynamiczny rozmiar batcha w
  ustalonym zakresie (1..`batch_size`).

- *Konfiguracja kwantyzatora:* do kwantyzacji użyto narzędzi z `torchao`
  (pipeline PT2E) oraz kwantyzatora `XNNPACKQuantizer` z `executorch`.
  Zastosowano globalną konfigurację symetrycznej kwantyzacji
  (`get_symmetric_quantization_config()`), ukierunkowaną na generowanie modelu
  kompatybilnego z backendem XNNPACK.

- *Wstawienie obserwatorów i kalibracja:* w kroku `prepare_*` do grafu
  wstawiane są elementy zbierające statystyki (obserwatory), które pozwalają
  wyznaczyć parametry kwantyzacji (m.in. skale). Następnie wykonano kalibrację,
  na wydzielonym zbiorze kalibracyjnym (ok. 10% zbioru treningowego).

- *Konwersja do modelu INT8:* po zakończeniu kalibracji wykonano
  `convert_pt2e(...)`, co zamienia odpowiednie operacje w grafie na ich
  odpowiedniki działające na reprezentacji skwantyzowanej (INT8), zgodnie z
  wcześniej wyznaczonymi parametrami.

Otrzymany model INT8 został następnie wykorzystany w dalszym etapie eksportu do
formatu `.pte`.

= Porównanie modeli

Porównanie wykonano w trzech obszarach: skuteczność klasyfikacji, rozmiar modelu oraz wydajność inferencji.

== Skuteczność (accuracy)

Dla zbioru testowego oba modele osiągają bardzo zbliżone wyniki - $91.87%$ dla FP32 i $91.76%$ dla INT8.
Model INT8 (@cm_int8) ma nieznacznie niższą średnią skuteczność niż FP32 (@cm_fp32)
(różnica jest mała i nie zmienia ogólnej jakości działania modelu).

#figure(
  image("img/fp32cm.png", height: 47%),
  caption: [Macierz pomyłek modelu FP32 (zbiór testowy)],
  placement: auto,
) <cm_fp32>

#figure(
  image("img/int8cm.png", height: 47%),
  caption: [Macierz pomyłek modelu INT8 (zbiór testowy)],
  placement: auto,
) <cm_int8>

Najczęściej występujące pomyłki powielają się dla modelu FP32 (@top-fp32) i INT8 (@top-int8).
Kwantyzacja wywołała niewielki zmiany wystąpień błędów (@cm_diff, @top-worst)

#let top_fp32 = csv("data/top_fp32.csv")
#let top_int8 = csv("data/top_int8.csv")
#let top_worst = csv("data/top_worst.csv")

#set par(justify: false)
#figure(
  table(
    columns: 3,
    [*Liczba wystąpień*],
    [*Prawdziwa Klasa*],
    [*Przewidziana Klasa*],
    ..top_fp32.flatten(),
  ),
  caption: [Najczęstsze błędy modelu FP32],
  placement: auto,
) <top-fp32>
#set par(justify: true)

#set par(justify: false)
#figure(
  table(
    columns: 3,
    [*Liczba wystąpień*],
    [*Prawdziwa Klasa*],
    [*Przewidziana Klasa*],
    ..top_int8.flatten(),
  ),
  caption: [Najczęstsze błędy modelu INT8],
  placement: auto,
) <top-int8>
#set par(justify: true)

#figure(
  image("img/diffcm.png", height: 47%),
  caption: [Macierz różnic (INT8 - FP32) dla zliczeń błędów; wartości dodatnie oznaczają częstsze występowanie pomyłki po kwantyzacji],
) <cm_diff>

#set par(justify: false)
#figure(
  table(
    columns: 5,
    [*Zmiana wystąpień błędów*],
    [*Prawdziwa Klasa*],
    [*Przewidziana Klasa*],
    [*Wystąpienia FP32*],
    [*Wystąpienia INT8*],
    ..top_worst.flatten(),
  ),
  caption: [Zmiana występowań błędów po kwantyzacji],
  placement: auto,
) <top-worst>
#set par(justify: true)

Na poziomie poszczególnych etykiet widać, że kwantyzacja nie wpływa jednakowo na wszystkie klasy (@per-label).
Mimo minimalnego spadku średniej accuracy,
model po kwantyzacji może lokalnie poprawiać rozpoznawanie niektórych etykiet, kosztem innyc.

#let per_label = csv("data/per_label.csv")

#set par(justify: false)
#set text(size: 0.75em)
#figure(
  table(
    columns: 8,
    [*Klasa*],
    [*Liczba sampli*],
    [*Poprawnych (FP32)*],
    [*%Poprawnych (FP32)*],
    [*Liczba sampli*],
    [*Poprawnych (INT8)*],
    [*%Poprawnych (INT8)*],
    [*%Różnica*],
    ..per_label.flatten(),
  ),
  caption: [Różnica skuteczności per-klasa],
  placement: auto,
) <per-label>
#set text(size: 1em)
#set par(justify: true)

== Rozmiar modelu

Rozmiar plików programu ExecuTorch (`.pte`) wynosił:
- FP32: 130 580 bajtów,
- INT8: 45 972 bajtów.

Oznacza to zmniejszenie rozmiaru o ok. 64.8% (model INT8 jest ok. 2.84x mniejszy od FP32).
Jest to istotne z punktu widzenia docelowego uruchamiania modelu w
aplikacji C++ oraz potencjalnego wdrażania na urządzeniach o ograniczonej
pamięci.

== Wydajność inferencji

Pomiary wydajności wykonano w aplikacji C++ z użyciem modeli:
- `tiny.pte` (FP32),
- `tiny_qt.pte` (INT8).

Test polegał na wykonaniu 50 rund rozgrzewki oraz 100000 inferencji na tym samym
wejściu (pojedynczy przykład danych) (@bench).
#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    inset: 4pt,

    [*Model*], [*min \[ns\]*], [*avg \[ns\]*], [*max \[ns\]*],

    [FP32 (`tiny.pte`)], [201 639], [216 591], [3 637 681],
    [INT8 (`tiny_qt.pte`)], [205 119], [217 850], [3 951 459],
  ),
  caption: [Porównanie wydajności],
) <bench>

Średni czas inferencji obu modeli jest bardzo zbliżony (ok. 0.217 ms na jedno
uruchomienie). W tym pomiarze model INT8 okazał się minimalnie wolniejszy
(około 1.26 µs różnicy w średniej), co może wynikać m.in. z narzutu backendu,
formatu wag/aktywacji albo sposobu partycjonowania grafu.

Warto podkreślić, że ten benchmark mierzy czas samej inferencji modelu dla
przygotowanego wejścia. W scenariuszu "na żywo" (mikrofon) całkowity koszt
systemu obejmuje również wyznaczenie cech (np. spektrogramu Mel), buforowanie
sygnału i obsługę I/O.

Mimo to, przy czasie rzędu 0.217 ms na inferencję, przetwarzanie w trybie
zbliżonym do real-time (np. przesuwające się okno) jest jak najbardziej
osiągalne na testowanej platformie: nawet dla kroku co 10 ms (100 inferencji/s)
czas samej inferencji stanowi niewielką część dostępnego budżetu czasowego.

= Problemy z Zephyrem i ExecuTorchem

Pirwotnym założeniem projektu było uruchomienie skwantyzowanego modelu na
płytce Arduino Nano 33 BLE Sense z systemem Zephyr i ExecuTorchem.
Płytka posiada mikrofon PDM, co pasowało do docelowego scenariusza "embedded".
Po wciśnięciu przycisku (GPIO) urządzenie miało nagrać 1 sekundę audio,
przekształcić sygnał do spektogramu, wykonać inferencję i wypisać wynik
na uarcie. Przetwarzanie sygnału miało być zaimplementowane z pomocą biblioteki
`cmsis-dsp`, co dało by znacznie lepsze rezultaty niż surowe C/C++ z powodu
wykorzystania akceleracji sprzętowej / operacji wektorowych. Dodatkowo
niektóre stałe (np. `hann_window`) miały być wyeksportowane z PyTorcha,
co też przyśpieszyło by ten proces.

W praktyce wsparcie ExecuTorcha na Zephyrze okazało się być bardzo esperymentalne
i w testowej konfiguracji nie udało się doprowadzić do poprawnej budowy aplikacji.
Próby były wykonywane na ExecuTorchu w wersji `a577584f927fe082256e4b8be7e2b9ada27c10f4`.

Napotkane problemy to m.in:

- brak dobrej dokumentacji: jedyny dowód, że takie coś jest w ogóle możliwe
  to przykład arm z ExecuTorch'a, który nie buduje się out-of-the-box
- problem z symlinkami: symlinki w `executorch/src` powodowały błędy;
  obejściem było ręczne ich zastąpienie hard linkami.
- problemy z backendem arm (głównie TOSA)
- problemy z vendorowanymi zależnościami, np. próby linkowania dynamicznego
  ze strony build systemu

W ostateczności plan uruchomienia na płytce wraz z Zephyrem został odłożony
z powodu braku czasu.

= Wnioski

- *Skuteczność (accuracy):* kwantyzacja spowodowała jedynie minimalny spadek
  średniej skuteczności na zbiorze testowym: 91.45% (FP32) → 91.16% (INT8),
  tj. różnica ok. \(-0.29\) pp. Jednocześnie wpływ kwantyzacji nie był
  równomierny dla wszystkich klas: dla części etykiet zanotowano niewielkie
  spadki (np. `zero`, `forward`, `visual`), ale dla innych pojawiły się
  poprawy (np. `go`, `sheila`, `off`, `eight`). Oznacza to, że INT8 nie jest "gorszy wszędzie",
  lecz zmienia rozkład błędów.
- *Charakter błędów:* analiza macierzy pomyłek pokazała, że najczęstsze pomyłki
  obu modeli są bardzo podobne (np. `forward -> four`, `off -> up`, `tree -> three`).
  Różnice dotyczą głównie liczności wybranych par błędów. Przykładowo,
  po kwantyzacji wzrosła liczba pomyłek `on -> off` (z 4 do 9).
  Takie przypadki są dobrym kandydatem do dalszej analizy (np. sprawdzenia,
  czy próbki są akustycznie podobne albo czy problem wynika z cech wejściowych).
- *Rozmiar modelu:* największą korzyścią kwantyzacji okazała się redukcja
  rozmiaru modelu `.pte`: 130 580 B (FP32) -> 45 972 B (INT8), czyli ok. 64.78%
  mniej (około 2.84×). Jest to istotne z punktu widzenia wdrożeń na urządzeniach
  o ograniczonej pamięci oraz dystrybucji modeli.
- *Możliwość przetwarzania w czasie rzeczywistym:* przy czasie inferencji rzędu
  ~0.217 ms na pojedyncze uruchomienie modelu, przetwarzanie z przesuwającym się
  oknem jest realne (np. przy kroku co 10 ms jest to ok. 100 inferencji/s, co
  nadal zostawia duży budżet czasowy). Należy jednak pamiętać, że w praktycznej
  aplikacji istotniejszy koszt najpewniej stanowić będzie przygotowanie cech
  (Mel-spektrogram) oraz obsługa buforowania i wejścia audio.

Podsumowując: kwantyzacja INT8 w badanym przypadku dała bardzo dużą redukcję
rozmiaru modelu przy niemal niezmienionej skuteczności, natomiast nie przyniosła
zauważalnego zysku szybkości w wykonanym benchmarku C++.

= Wykorzystane źródła / biblioteki

- SPEECHCOMMANDS dataset - https://huggingface.co/datasets/google/speech_commands
- Pierwotne źródło modelu - https://www.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html
- PyTorch (v2.9.0) - https://github.com/pytorch/pytorch
- TorchAudio (v2.9.0) - https://github.com/pytorch/audio
- ExecuTorch (v1.0.1) - https://github.com/pytorch/executorch
- cxxopts (v3.3.1) - https://github.com/jarro2783/cxxopts
