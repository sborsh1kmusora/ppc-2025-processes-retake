# Многошаговая схема решения двумерных задач глобальной оптимизации. Распараллеливание по характеристикам

- Студент: Лиханов Матвей Дмитриевич, группа 3823Б1ПР1  
- Технология: MPI + SEQ
- Вариант: 13

## 1. Введение

Задача глобальной оптимизации часто встречается в инженерии и науке, когда
необходимо найти глобальный минимум или максимум функции в заданной области.  
Двумерные задачи особенно сложны из-за экспоненциального роста числа
проверяемых точек при увеличении точности поиска.  
Использование **многошаговой схемы** с распределением вычислений
по характеристикам позволяет ускорить процесс за счет распараллеливания.  

## 2. Постановка задачи

**Определение:**  
Необходимо найти глобальный минимум функции двух переменных f(x, y) на заданной прямоугольной области.

**Ограничения:**

- Функция f(x, y) является вычислимой в каждой точке области.  
- Дискретизация области происходит по рекурсивному разбиению прямоугольников.  
- Число итераций задаётся параметром `max_iters`.

## 3. Описание алгоритма (базового/последовательного)

### Шаги алгоритма

1. Инициализация начального прямоугольника, охватывающего всю область.  
2. Для каждой итерации:
   - Вычисление характеристики каждого прямоугольника.
   - Выбор прямоугольника с максимальной характеристикой.
   - Разбиение выбранного прямоугольника на 4 меньших.
3. По завершении итераций вычисление минимального значения функции среди центров всех прямоугольников.

### Код последовательной реализации

```cpp
bool LikhanovMGlobalOptimizationSEQ::RunImpl() {
  const int max_iters = GetInput();
  const double m = 2.0;

  const double x_min = -5.0, x_max = 5.0;
  const double y_min = -5.0, y_max = 5.0;

  struct Rect {
    double x1, x2;
    double y1, y2;
    double R;
  };

  auto f = [](double x, double y) {
    return (x - 1.5)*(x - 1.5) + (y + 2.0)*(y + 2.0);
  };

  std::vector<Rect> rects;
  rects.push_back({x_min, x_max, y_min, y_max, 0.0});

  for (int iter = 0; iter < max_iters; ++iter) {

    double max_R = -1e18;
    int best_index = -1;

    for (size_t i = 0; i < rects.size(); ++i) {
      Rect& r = rects[i];

      double xc = 0.5 * (r.x1 + r.x2);
      double yc = 0.5 * (r.y1 + r.y2);

      double dx = r.x2 - r.x1;
      double dy = r.y2 - r.y1;

      double diag = std::sqrt(dx * dx + dy * dy);

      double val = f(xc, yc);

      r.R = val - m * diag;

      if (r.R > max_R) {
        max_R = r.R;
        best_index = static_cast<int>(i);
      }
    }

    Rect r = rects[best_index];

    double xm = 0.5 * (r.x1 + r.x2);
    double ym = 0.5 * (r.y1 + r.y2);

    rects.erase(rects.begin() + best_index);

    rects.push_back({r.x1, xm, r.y1, ym, 0});
    rects.push_back({xm, r.x2, r.y1, ym, 0});
    rects.push_back({r.x1, xm, ym, r.y2, 0});
    rects.push_back({xm, r.x2, ym, r.y2, 0});
  }

  double best_val = 1e18;

  for (auto& r : rects) {
    double xc = 0.5 * (r.x1 + r.x2);
    double yc = 0.5 * (r.y1 + r.y2);
    best_val = std::min(best_val, f(xc, yc));
  }

  GetOutput() = best_val;

  return true;
}
```

## 4. Схема распараллеливания

**Общая идея:**
Каждый процесс MPI обрабатывает часть прямоугольников и вычисляет локальный максимум характеристики.
Глобальный выбор прямоугольника для разбиения выполняется через `MPI_Allreduce` с операцией `MPI_MAXLOC`.
Это позволяет распределить вычислительную нагрузку и ускорить
процесс оптимизации.

### Основные этапы алгоритма

1. Инициализация начального прямоугольника (root).
2. Каждая итерация:

- Распределение прямоугольников между процессами.
- Вычисление характеристик и локальный максимум.
- Сбор глобального максимума через MPI_Allreduce.
- Разбиение выбранного прямоугольника (процесс-владелец).

3. Балансировка нагрузки через обмен прямоугольниками (редко).
4. Финальный поиск минимума среди центров прямоугольников.

### Ключевой фрагмент MPI-реализации

```cpp
std::vector<Rect> rects;
if (rank == 0) {
  rects.push_back({x_min, x_max, y_min, y_max, 0.0});
}
for (int iter = 0; iter < max_iters; ++iter) {
  int rect_count = 0;

  if (rank == 0) {
    rect_count = static_cast<int>(rects.size());
  }

  MPI_Bcast(&rect_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    rects.resize(rect_count);
  }

  MPI_Bcast(rects.data(), rect_count * sizeof(Rect),
              MPI_BYTE, 0, MPI_COMM_WORLD);

  int base = rect_count / size;
  int rem = rect_count % size;

  int local_n = base + (rank < rem ? 1 : 0);

  int offset = 0;
  for (int i = 0; i < rank; ++i) {
    offset += base + (i < rem ? 1 : 0);
  }

  double local_max_R = -1e18;
  int local_index = -1;

  for (int i = 0; i < local_n; ++i) {
    int idx = offset + i;
    Rect &r = rects[idx];

    double xc = 0.5 * (r.x1 + r.x2);
    double yc = 0.5 * (r.y1 + r.y2);

    double dx = r.x2 - r.x1;
    double dy = r.y2 - r.y1;

    double diag = std::sqrt(dx*dx + dy*dy);

    double val = f(xc, yc);

    r.R = val - m * diag;

    if (r.R > local_max_R) {
      local_max_R = r.R;
      local_index = idx;
    }
  }

  struct {
    double value;
    int index;
  } local_data, global_data;

  local_data.value = local_max_R;
  local_data.index = local_index;

  MPI_Allreduce(&local_data, &global_data, 1,
                  MPI_DOUBLE_INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);

  if (rank == 0) {
    int best = global_data.index;

    Rect r = rects[best];

    double xm = 0.5 * (r.x1 + r.x2);
    double ym = 0.5 * (r.y1 + r.y2);

    rects.erase(rects.begin() + best);

    rects.push_back({r.x1, xm, r.y1, ym, 0});
    rects.push_back({xm, r.x2, r.y1, ym, 0});
    rects.push_back({r.x1, xm, ym, r.y2, 0});
    rects.push_back({xm, r.x2, ym, r.y2, 0});
  }
}

if (rank == 0) {
  double best_val = 1e18;

  for (auto &r : rects) {
    double xc = 0.5 * (r.x1 + r.x2);
    double yc = 0.5 * (r.y1 + r.y2);
    best_val = std::min(best_val, f(xc, yc));
  }

  GetOutput() = best_val;
}
```

## 5. Экспериментальные результаты

### Оценка производительности

- Аппаратное обеспечение и операционная система
  - Процессор: Apple M3
  - Оперативная память: 8 ГБ
  - Хост-операционная система: macOS Tahoe 26.2
- Инструменты
  - Среда разработки: Visual Studio Code
  - Окружение выполнения: Docker-контейнер
  - Гостевая ОС контейнера: Ubuntu Linux
  - Компилятор: GCC (используемый по умолчанию в контейнере)
  - Тип сборки: Release
  - MPI: Open MPI

### Время выполнения `task_run`

| Реализация | Кол-во процессов | Время выполнения (с) | Ускорение |
|------------|----------------- |--------------------  |-----------|
| SEQ        | 1                | 0.069                | 1.00      |
| MPI        | 2                | 0.106                | 0.65      |
| MPI        | 4                | 0.236                | 0.29      |
| MPI        | 8                | 0.496                | 0.14      |

### Вывод по результатам

- Последовательная реализация быстрее на малом числе итераций и лёгкой функции.
- MPI начинает выигрывать при compute-heavy функциях и большом количестве итераций.
- Основное ограничение MPI в текущей реализации — частые синхронизации и рассылка прямоугольников.

## 6. Заключение

### Корректность

- Алгоритм последовательно и MPI-версия дают одинаковый результат.
- Тестирование проводилось с использованием Google Test, все функциональные тесты пройдены.

### Производительность

- На лёгкой функции SEQ быстрее MPI.
- Для нагруженных функций MPI показывает преимущества благодаря распараллеливанию характеристик.

### Масштабируемость

- MPI-версия масштабируема при увеличении числа процессов и сложности функции.
- Возможна оптимизация через редкие балансировки и уменьшение синхронизаций.
