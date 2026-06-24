import type { ChatMessage } from "./types";

export type EffortField = "thinking_effort" | "reasoning_effort";
export type EffortLevel = "" | "low" | "medium" | "high" | "xhigh";

export const CHAT_COMPLETIONS_PATH = "/v1/chat/completions";

type BuildChatCompletionRequestBodyArgs = {
  model: string;
  messages: ChatMessage[];
  effortField: EffortField;
  effortLevel: EffortLevel;
};

export type ChatCompletionRequestBody = {
  model: string;
  messages: ChatMessage[];
  thinking_effort?: Exclude<EffortLevel, "">;
  reasoning_effort?: Exclude<EffortLevel, "">;
};

export type ChatCompletionRequest = {
  path: typeof CHAT_COMPLETIONS_PATH;
  options: {
    method: "POST";
    body: ChatCompletionRequestBody;
  };
};

export function buildChatCompletionRequestBody({
  model,
  messages,
  effortField,
  effortLevel,
}: BuildChatCompletionRequestBodyArgs): ChatCompletionRequestBody {
  const body: ChatCompletionRequestBody = {
    model: model.trim() || "auto",
    messages,
  };

  if (effortLevel) {
    body[effortField] = effortLevel;
  }

  return body;
}

export function buildChatCompletionRequest(args: BuildChatCompletionRequestBodyArgs): ChatCompletionRequest {
  return {
    path: CHAT_COMPLETIONS_PATH,
    options: {
      method: "POST",
      body: buildChatCompletionRequestBody(args),
    },
  };
}
