"""Reference resolvers for @ syntax."""

from msagent.cli.resolvers.base import RefType, Resolver
from msagent.cli.resolvers.file import FileResolver
from msagent.cli.resolvers.image import ImageResolver

__all__ = ["FileResolver", "ImageResolver", "RefType", "Resolver"]
