# Распознавание русского жестового языка

Модель распознавания изолированных жестов русского жестового языка (РЖЯ) на основе скелетных координат кистей рук.

## Результаты

| Подвыборка | Top-1 Accuracy | F1 (macro) | Top-5 Accuracy |
|------------|-----------|------------|-----------|
| Train | 66.96% | 66.29% | 92.92% |
| Validation | 65.22% | 63.54% | 90.33% |
| **Test** | **50.5%** | **49.05%** | **80.05%** |

## Ссылки

| Ресурс | Ссылка |
|--------|--------|
| **Датасет Slovo** | [GitHub Slovo](https://github.com/hukenovs/slovo) |
| **Веса модели** | [Скачать best_model.pt]([https://drive.google.com/your-link-here](https://drive.google.com/uc?export=download&id=1rc1pexeKRCYHHZmLQttR-lsGvap8gsJI)) |

### Цитирование датасета

Если вы используете датасет Slovo в своих исследованиях, пожалуйста, ссылайтесь на оригинальную работу:

```bibtex
@inproceedings{kapitanov2023slovo,
    title={Slovo: Russian Sign Language Dataset},
    author={Kapitanov, Alexander and Karina, Kvanchiani and Nagaev, Alexander and Elizaveta, Petrova},
    booktitle={International Conference on Computer Vision Systems},
    pages={63--73},
    year={2023},
    organization={Springer}
}
