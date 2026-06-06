// SPDX-License-Identifier: Apache-2.0
/**
 * Russian (ru) message catalog.
 *
 * Translations for sndr Control Center. Maintained by translators via
 * the .po workflow. Keys 1:1 with en/messages.ts.
 */
export const messages = {
  // ── Navigation ────────────────────────────────────────────────────────
  'nav.group.live': 'Живое',
  'nav.group.engines': 'Движки',
  'nav.group.workloads': 'Нагрузка',
  'nav.group.health': 'Здоровье',
  'nav.group.admin': 'Администрирование',
  'nav.overview': 'Обзор',
  'nav.fleet': 'Флот',
  'nav.hosts': 'Хосты',
  'nav.containers': 'Контейнеры',
  'nav.engines': 'Движки',
  'nav.pins': 'Пины',
  'nav.patches': 'Патчи',
  'nav.drift': 'Расхождения',
  'nav.bench': 'Бенчи',
  'nav.chat': 'Чат',
  'nav.jobs': 'Задачи',
  'nav.doctor': 'Доктор',
  'nav.evidence': 'Доказательства',
  'nav.configs': 'Конфигурации',
  'nav.licensing': 'Лицензирование',
  'nav.auth': 'Аутентификация',
  'nav.settings': 'Настройки',

  // ── Common ────────────────────────────────────────────────────────────
  'common.loading': 'Загрузка…',
  'common.error': 'Произошла ошибка',
  'common.retry': 'Повторить',
  'common.cancel': 'Отмена',
  'common.confirm': 'Подтвердить',
  'common.save': 'Сохранить',
  'common.close': 'Закрыть',
  'common.clear': 'Очистить',
  'common.send': 'Отправить',
  'common.no_data': 'Нет данных',
  'common.last_run': 'Последний запуск',
  'common.summary': 'Сводка',
  'common.timestamp': 'Время',
  'common.model': 'Модель',
  'common.pin': 'Пин',
  'common.engine': 'Движок',

  // ── Status / tags ─────────────────────────────────────────────────────
  'status.applied': 'Применено',
  'status.skipped': 'Пропущено',
  'status.failed': 'Не удалось',
  'status.drift': 'Обнаружено расхождение',
  'status.ok': 'OK',
  'status.online': 'В сети',
  'status.offline': 'Не в сети',
  'status.degraded': 'Деградация',
  'status.unknown': 'Неизвестно',
  'status.running': 'Работает',
  'status.exited': 'Завершён',
  'status.paused': 'Приостановлен',
  'status.created': 'Создан',
  'status.dead': 'Мёртв',
  'status.restarting': 'Перезапуск',
  'status.queued': 'В очереди',
  'status.succeeded': 'Успешно',
  'status.canceled': 'Отменён',
  'status.pending': 'Ожидание',
  'status.regression': 'Регрессия',
  'status.warning': 'Предупреждение',
  'status.info': 'Информация',
  'status.error': 'Ошибка',
  'status.critical': 'Критично',

  // ── Tiers ─────────────────────────────────────────────────────────────
  'tier.community': 'Сообщество',
  'tier.engine': 'Engine (коммерческий)',

  // ── Overview ──────────────────────────────────────────────────────────
  'overview.title': 'Обзор',
  'overview.fleet.title': 'Флот',
  'overview.containers.title': 'Контейнеры',
  'overview.patches.title': 'Патчи',
  'overview.doctor.title': 'Доктор',
  'overview.evidence.title': 'Доказательства',
  'overview.api_version': 'Версия API',

  // ── Fleet ─────────────────────────────────────────────────────────────
  'fleet.title': 'Флот',
  'fleet.total_hosts': 'Всего хостов',
  'fleet.online': 'В сети',
  'fleet.degraded': 'Деградация',
  'fleet.offline': 'Не в сети',
  'fleet.unknown': 'Неизвестно',
  'fleet.gpus': 'GPU',
  'fleet.vram': 'VRAM (GiB)',

  // ── Hosts ─────────────────────────────────────────────────────────────
  'hosts.title': 'Хосты',
  'hosts.empty': 'Во флоте нет хостов',
  'hosts.col.hostname': 'Имя хоста',
  'hosts.col.status': 'Статус',
  'hosts.col.sndr_version': 'sndr',
  'hosts.col.engine': 'Движок',
  'hosts.col.gpus': 'GPU',
  'hosts.col.ram': 'RAM',

  // ── Containers ────────────────────────────────────────────────────────
  'containers.title': 'Контейнеры',
  'containers.empty': 'Контейнеры не запущены',
  'containers.col.name': 'Имя',
  'containers.col.state': 'Состояние',
  'containers.col.image': 'Образ',
  'containers.col.model': 'Модель',
  'containers.col.engine': 'Движок',
  'containers.col.ports': 'Порты',

  // ── Engines ───────────────────────────────────────────────────────────
  'engines.title': 'Движки',
  'engines.selector.title': 'Движок',
  'engines.selector.placeholder': 'Выберите движок',
  'engines.unavailable.suffix': '(недоступен)',

  // ── Pins ──────────────────────────────────────────────────────────────
  'pins.title': 'Пины',
  'pins.empty': 'Манифесты пинов отсутствуют',
  'pins.col.pin': 'Пин',
  'pins.col.status': 'Статус',
  'pins.col.full_version': 'Полная версия',
  'pins.col.manifest': 'Манифест',
  'pins.col.drift': 'Расхождение',
  'pins.col.bench': 'Бенч (TPS)',

  // ── Patches ───────────────────────────────────────────────────────────
  'patches.title': 'Патчи',
  'patches.inventory.title': 'Инвентарь',
  'patches.inventory.total': 'Всего',
  'patches.inventory.active': 'Активных',
  'patches.inventory.retired': 'Снятых',
  'patches.inventory.enabled_now': 'Включено сейчас',
  'patches.inventory.engine_tier': 'Engine tier',
  'patches.filter.family': 'Семейство',
  'patches.filter.tier': 'Tier',
  'patches.filter.lifecycle': 'Жизненный цикл',
  'patches.filter.enabled_only': 'Только включённые в окружении',
  'patches.col.id': 'ID',
  'patches.col.title': 'Название',
  'patches.col.family': 'Семейство',
  'patches.col.tier': 'Tier',
  'patches.col.lifecycle': 'Жизненный цикл',
  'patches.col.default_on': 'По умолчанию',
  'patches.col.enabled_now': 'Включён',

  // ── Drift ─────────────────────────────────────────────────────────────
  'drift.title': 'Расхождения',

  // ── Bench ─────────────────────────────────────────────────────────────
  'bench.title': 'Бенчи',
  'bench.empty': 'Бенчей пока нет',
  'bench.col.wall_tps': 'wall TPS',
  'bench.col.tpot': 'TPOT мс',
  'bench.col.ttft': 'TTFT мс',
  'bench.col.cv': 'CV',
  'bench.col.outcome': 'Результат',

  // ── Chat ──────────────────────────────────────────────────────────────
  'chat.title': 'Чат',
  'chat.endpoint': 'Эндпоинт',
  'chat.model': 'Модель',
  'chat.placeholder': 'Введите сообщение…',
  'chat.send_hint': 'Отправить (⌘+⏎)',
  'chat.no_messages': 'Сообщений пока нет. Введите ниже, чтобы начать.',
  'chat.generating': 'Генерация…',

  // ── Jobs ──────────────────────────────────────────────────────────────
  'jobs.title': 'Задачи',
  'jobs.empty': 'Нет задач',
  'jobs.col.id': 'ID',
  'jobs.col.kind': 'Тип',
  'jobs.col.state': 'Состояние',
  'jobs.col.progress': 'Прогресс',
  'jobs.col.started': 'Начало',

  // ── Doctor ────────────────────────────────────────────────────────────
  'doctor.title': 'Доктор',
  'doctor.healthy': 'Все здорово',
  'doctor.issues': 'Найдены проблемы',
  'doctor.no_findings': 'Замечаний нет (всё зелёное)',
  'doctor.col.severity': 'Уровень',
  'doctor.col.category': 'Категория',
  'doctor.col.finding': 'Заголовок',
  'doctor.col.remediation': 'Рекомендация',

  // ── Evidence ──────────────────────────────────────────────────────────
  'evidence.title': 'Доказательства',
  'evidence.release_readiness': 'Готовность к релизу',
  'evidence.col.id': 'ID',
  'evidence.col.name': 'Гейт',
  'evidence.col.status': 'Статус',

  // ── Configs ───────────────────────────────────────────────────────────
  'configs.title': 'Конфигурации',
  'configs.tab.models': 'Модели',
  'configs.tab.hardware': 'Оборудование',
  'configs.tab.profiles': 'Профили',
  'configs.tab.presets': 'Пресеты',

  // ── Licensing ─────────────────────────────────────────────────────────
  'licensing.title': 'Лицензирование',

  // ── Auth ──────────────────────────────────────────────────────────────
  'auth.title': 'Аутентификация',
  'auth.session_status': 'Статус сессии',
  'auth.authenticated': 'Аутентифицирован',
  'auth.token_preview': 'Превью токена',
  'auth.api_token': 'API-токен',
  'auth.token_placeholder': 'sndr_…',
  'auth.token_label': 'Bearer-токен для API',

  // ── Settings ──────────────────────────────────────────────────────────
  'settings.title': 'Настройки',
  'settings.locale': 'Локаль',
  'settings.locale.label': 'Язык интерфейса',
  'settings.engine': 'Активный движок',
  'settings.api_endpoint': 'Эндпоинт API',
  'settings.api_endpoint.label': 'Переопределить базовый URL (пусто = тот же origin)',
  'settings.about': 'О программе',
  'settings.version': 'Версия',
  'settings.theme': 'Тема',
  'settings.i18n': 'i18n',
} as const;

export type MessageKey = keyof typeof messages;
