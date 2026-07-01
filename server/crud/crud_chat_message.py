from server.crud.base import CRUDBase
from server.models.genai import ChatMessage
from server.schemas.genai import ChatMessageCreate, ChatMessageUpdate


class CRUDChatMessage(CRUDBase[ChatMessage, ChatMessageCreate, ChatMessageUpdate]):
    def create(self, db, *, obj_in: ChatMessageCreate) -> ChatMessage:
        return super().create(db, obj_in=obj_in)


chat_message = CRUDChatMessage(ChatMessage)
