import {
  Alert,
  Button,
  Card,
  Checkbox,
  Group,
  LoadingOverlay,
  MultiSelect,
  Progress,
  Select,
  Stack,
  Text,
  Textarea,
} from '@mantine/core';
import { useEffect, useMemo, useState } from 'react';

import type { ParticipantData } from '../../../storage/types';
import { useStorageEngine } from '../../../storage/storageEngineHooks';

type GeminiAnalyzeResponse = {
  summary?: string;
  raw?: unknown;
};

type GeminiRestResponse = {
  candidates?: Array<{
    content?: {
      parts?: Array<{ text?: string }>;
    };
  }>;
  text?: string;
};

type RecordingItem = {
  participantId: string;
  componentName: string;
  trialOrder: string;
  identifier: string; // `${componentName}_${trialOrder}`
};

function extractGeminiText(json: unknown): string | null {
  const data = json as GeminiRestResponse;
  const first = Array.isArray(data.candidates) && data.candidates.length > 0 ? data.candidates[0] : null;
  const parts = first?.content?.parts;
  const textParts = Array.isArray(parts)
    ? parts
      .map((p) => (typeof p?.text === 'string' ? p.text : null))
      .filter((t): t is string => t !== null)
    : [];

  if (textParts.length > 0) return textParts.join('');

  const fallbackText = typeof data.text === 'string' ? data.text : undefined;
  if (typeof fallbackText === 'string') return fallbackText;
  return null;
}

async function fetchBlobFromObjectUrl(url: string) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch recording bytes (status ${res.status})`);
  }
  return res.blob();
}

function getDefaultPrompt() {
  return [
    'Provide a high-level summary of the screen recording in 3-5 sentences.',
    'Include key events and, when possible, reference moments using MM:SS timestamps (e.g., 01:15).',
    'If you detect the participant struggling or changing strategy, mention that explicitly.',
  ].join(' ');
}

export function MassScreenRecordingSummarizationView({
  visibleParticipants,
}: {
  visibleParticipants: ParticipantData[];
}) {
  const { storageEngine } = useStorageEngine();

  const env = import.meta.env as unknown as Record<string, string | undefined>;
  const apiKey = env.VITE_GEMINI_API_KEY;
  const model = env.VITE_GEMINI_VIDEO_MODEL;
  const effectiveModel = model || 'models/gemini-2.0-flash';

  const [participantIdFilter, setParticipantIdFilter] = useState<string>('ALL');
  const [trialOrderFilter, setTrialOrderFilter] = useState<string[]>([]);

  const [prompt, setPrompt] = useState<string>(getDefaultPrompt());
  const [skipIfExists, setSkipIfExists] = useState<boolean>(true);
  const [persistResults, setPersistResults] = useState<boolean>(true);

  const maxInlineBytes = 20 * 1024 * 1024;

  const baseRecordings = useMemo(() => {
    const items: RecordingItem[] = [];

    for (const participant of visibleParticipants) {
      const uniqueWithinParticipant = new Map<string, RecordingItem>();
      for (const a of Object.values(participant.answers)) {
        if (a.endTime > 0) {
          const identifier = `${a.componentName}_${a.trialOrder}`;
          uniqueWithinParticipant.set(identifier, {
            participantId: participant.participantId,
            componentName: a.componentName,
            trialOrder: a.trialOrder,
            identifier,
          });
        }
      }

      items.push(...Array.from(uniqueWithinParticipant.values()));
    }

    return items;
  }, [visibleParticipants]);

  const filteredRecordings = useMemo(() => {
    if (participantIdFilter === 'ALL' && trialOrderFilter.length === 0) return baseRecordings;

    const trialOrderSet = new Set(trialOrderFilter);
    return baseRecordings.filter((r) => (
      (participantIdFilter === 'ALL' || r.participantId === participantIdFilter)
      && (trialOrderFilter.length === 0 || trialOrderSet.has(r.trialOrder))
    ));
  }, [baseRecordings, participantIdFilter, trialOrderFilter]);

  const filteredKeys = useMemo(
    () => new Set(filteredRecordings.map((r) => `${r.participantId}::${r.identifier}`)),
    [filteredRecordings],
  );

  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(() => new Set());

  const selectedCount = useMemo(() => {
    let count = 0;
    selectedKeys.forEach((k) => {
      if (filteredKeys.has(k)) count += 1;
    });
    return count;
  }, [selectedKeys, filteredKeys]);

  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 });
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, { summary?: string; status: 'ok' | 'skipped' | 'failed' | 'too_large' }>>({});

  useEffect(() => {
    // If filters change, we keep selection for those items still visible; but prune keys that are no longer in view.
    setSelectedKeys((prev) => new Set(Array.from(prev).filter((k) => filteredKeys.has(k))));
  }, [filteredKeys]);

  const selectionToggle = (key: string, checked: boolean) => {
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (checked) next.add(key);
      else next.delete(key);
      return next;
    });
  };

  const selectAll = () => {
    setSelectedKeys(new Set(filteredRecordings.map((r) => `${r.participantId}::${r.identifier}`)));
  };

  const clearSelection = () => setSelectedKeys(new Set());

  const uniqueParticipantIds = useMemo(
    () => Array.from(new Set(visibleParticipants.map((p) => p.participantId))).sort(),
    [visibleParticipants],
  );

  const uniqueTrialOrders = useMemo(
    () => Array.from(new Set(baseRecordings.map((r) => r.trialOrder))).sort((a, b) => {
      const na = Number(a);
      const nb = Number(b);
      if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
      return a.localeCompare(b);
    }),
    [baseRecordings],
  );

  const analyzeInline = async (file: File) => {
    if (!apiKey) throw new Error('Missing VITE_GEMINI_API_KEY');

    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('Failed to read video file'));
      reader.onload = () => resolve(String(reader.result));
      reader.readAsDataURL(file);
    });

    const dataMatch = dataUrl.match(/^data:(.+);base64,(.+)$/);
    if (!dataMatch) {
      throw new Error('Invalid base64 data URL');
    }

    const mimeType = dataMatch[1];
    const base64Data = dataMatch[2];

    const url = `https://generativelanguage.googleapis.com/v1beta/${effectiveModel}:generateContent?key=${encodeURIComponent(apiKey)}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                inlineData: {
                  mimeType,
                  data: base64Data,
                },
              },
              { text: prompt },
            ],
          },
        ],
      }),
    });

    const json = await res.json().catch(() => ({}));
    if (!res.ok) {
      return { summary: undefined, raw: { status: res.status, json } } as GeminiAnalyzeResponse;
    }

    const text = extractGeminiText(json);
    return { summary: text ?? undefined, raw: json } as GeminiAnalyzeResponse;
  };

  const analyzeSelected = async () => {
    if (!storageEngine) return;
    if (!apiKey) {
      setError('Missing VITE_GEMINI_API_KEY');
      return;
    }
    if (selectedCount === 0) return;

    setIsRunning(true);
    setError(null);
    setProgress({ done: 0, total: selectedCount });

    const selectedItems = filteredRecordings.filter((r) => selectedKeys.has(`${r.participantId}::${r.identifier}`));

    setResults((prev) => {
      const entries = selectedItems.map((i) => [`${i.participantId}::${i.identifier}`, { status: 'ok' as const }]);
      return { ...prev, ...Object.fromEntries(entries) };
    });

    const processOne = async (item: RecordingItem) => {
      const itemKey = `${item.participantId}::${item.identifier}`;

      try {
        if (skipIfExists) {
          const existingSummaryUrl = await storageEngine.getScreenRecordingSummary(item.identifier, item.participantId);
          if (existingSummaryUrl) {
            URL.revokeObjectURL(existingSummaryUrl);
            setResults((prev) => ({ ...prev, [itemKey]: { status: 'skipped' } }));
            setProgress((p) => ({ ...p, done: p.done + 1 }));
            return;
          }
        }

        const videoObjectUrl = await storageEngine.getScreenRecording(item.identifier, item.participantId);
        if (!videoObjectUrl) {
          setResults((prev) => ({ ...prev, [itemKey]: { status: 'failed' } }));
          setProgress((p) => ({ ...p, done: p.done + 1 }));
          return;
        }

        const blob = await fetchBlobFromObjectUrl(videoObjectUrl);
        URL.revokeObjectURL(videoObjectUrl);

        if (blob.size > maxInlineBytes) {
          setResults((prev) => ({ ...prev, [itemKey]: { status: 'too_large' } }));
          setProgress((p) => ({ ...p, done: p.done + 1 }));
          return;
        }

        const file = new File([blob], `${item.identifier}.webm`, { type: blob.type || 'video/webm' });
        const result = await analyzeInline(file);

        if (!result.summary) {
          setResults((prev) => ({ ...prev, [itemKey]: { status: 'failed' } }));
          setProgress((p) => ({ ...p, done: p.done + 1 }));
          return;
        }

        setResults((prev) => ({
          ...prev,
          [itemKey]: { status: 'ok', summary: result.summary },
        }));

        if (persistResults) {
          const summaryBlob = new Blob([
            JSON.stringify(
              {
                summary: result.summary,
                prompt,
                model: effectiveModel,
              },
              null,
              2,
            ),
          ], { type: 'application/json' });
          await storageEngine.saveScreenRecordingSummary(summaryBlob, item.identifier, item.participantId);
        }

        setProgress((p) => ({ ...p, done: p.done + 1 }));
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Failed to summarize clip';
        setResults((prev) => ({ ...prev, [itemKey]: { status: 'failed' } }));
        setError(msg);
        setProgress((p) => ({ ...p, done: p.done + 1 }));
      }
    };

    try {
      await selectedItems.reduce(
        (prevPromise, item) => prevPromise.then(() => processOne(item)),
        Promise.resolve(),
      );
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <Card withBorder shadow="sm" padding="md">
      <Stack gap="md">
        <Group justify="space-between">
          <div>
            <Text fw={700}>Mass screen recording summarization</Text>
            <Text size="sm" color="dimmed">
              Select multiple stored clips across the study, edit the prompt, then summarize them in bulk.
            </Text>
          </div>
          <Group gap="xs">
            <Button variant="default" onClick={selectAll} disabled={filteredRecordings.length === 0 || isRunning}>
              Select all
            </Button>
            <Button variant="default" onClick={clearSelection} disabled={isRunning}>
              Clear
            </Button>
          </Group>
        </Group>

        <Stack gap="xs">
          <Group>
            <Select
              label="Participant filter"
              data={[
                { value: 'ALL', label: 'All participants' },
                ...uniqueParticipantIds.map((id) => ({ value: id, label: id })),
              ]}
              value={participantIdFilter}
              onChange={(v) => setParticipantIdFilter(v || 'ALL')}
              disabled={isRunning}
            />

            <MultiSelect
              label="Trial filter"
              data={uniqueTrialOrders.map((t) => ({ value: t, label: `Trial ${t}` }))}
              value={trialOrderFilter}
              onChange={setTrialOrderFilter}
              placeholder="All trials"
              disabled={isRunning}
            />
          </Group>
        </Stack>

        {!apiKey && (
          <Alert title="Missing API key" color="red" variant="light">
            This needs
            <code>VITE_GEMINI_API_KEY</code>
            {' '}
            set in your environment
            {' '}
            (Vite client env).
          </Alert>
        )}

        <Textarea
          minRows={3}
          autosize
          label="Gemini prompt (applied to every selected recording)"
          value={prompt}
          onChange={(e) => setPrompt(e.currentTarget.value)}
          disabled={isRunning}
        />

        <Group>
          <Checkbox
            label="Skip clips that already have a stored summary"
            checked={skipIfExists}
            onChange={(e) => setSkipIfExists(e.currentTarget.checked)}
            disabled={isRunning}
          />
          <Checkbox
            label="Persist new summaries back into the study"
            checked={persistResults}
            onChange={(e) => setPersistResults(e.currentTarget.checked)}
            disabled={isRunning}
          />
        </Group>

        <Stack gap="sm">
          <Group justify="space-between">
            <Text size="sm" color="dimmed">
              Clips found:
              {' '}
              {filteredRecordings.length}
              .
              {' '}
              Selected:
              {' '}
              {selectedCount}
              .
            </Text>
            {selectedCount > 0 && (
              <Button
                onClick={analyzeSelected}
                loading={isRunning}
                disabled={!apiKey || isRunning || storageEngine === undefined}
              >
                Analyze selected
                {' '}
                (
                {selectedCount}
                )
              </Button>
            )}
          </Group>

          <LoadingOverlay visible={isRunning} overlayProps={{ blur: 2 }} />

          <Stack gap="xs" style={{ maxHeight: 360, overflow: 'auto' }}>
            {filteredRecordings.length === 0 ? (
              <Alert color="yellow" variant="light">
                No screen-recorded clips were found for the selected filters.
              </Alert>
            ) : (
              filteredRecordings.map((r) => {
                const key = `${r.participantId}::${r.identifier}`;
                const isChecked = selectedKeys.has(key);
                const result = results[key];
                return (
                  <Card
                    key={key}
                    withBorder
                    shadow="xs"
                    padding="sm"
                    style={{ background: isChecked ? 'rgba(0, 0, 0, 0.02)' : undefined }}
                  >
                    <Group justify="space-between">
                      <Checkbox
                        checked={isChecked}
                        onChange={(e) => selectionToggle(key, e.currentTarget.checked)}
                        disabled={isRunning}
                        label={`${r.componentName} (trial ${r.trialOrder})`}
                      />
                      <Text size="xs" color="dimmed">
                        {r.participantId}
                      </Text>
                    </Group>

                    {result?.status && (
                      <Text
                        size="xs"
                        mt={4}
                        color={result.status === 'ok' ? 'green' : result.status === 'skipped' ? 'blue' : 'red'}
                      >
                        {result.status === 'ok'
                          ? 'Ready'
                          : result.status === 'skipped'
                            ? 'Skipped (already summarized)'
                            : result.status === 'too_large'
                              ? `Too large (> ${maxInlineBytes / (1024 * 1024)}MB)`
                              : 'Failed'}
                      </Text>
                    )}

                    {result?.summary && (
                      <Text size="sm" mt="xs" style={{ whiteSpace: 'pre-wrap' }}>
                        {result.summary}
                      </Text>
                    )}
                  </Card>
                );
              })
            )}
          </Stack>
        </Stack>

        {isRunning && progress.total > 0 && (
          <>
            <Progress value={(progress.done / progress.total) * 100} />
            <Text size="sm" color="dimmed">
              {progress.done}
              {' '}
              /
              {' '}
              {progress.total}
              {' '}
              summarized
            </Text>
          </>
        )}

        {error && (
          <Alert color="red" variant="light">
            {error}
          </Alert>
        )}
      </Stack>
    </Card>
  );
}
// end of file
// EOF
