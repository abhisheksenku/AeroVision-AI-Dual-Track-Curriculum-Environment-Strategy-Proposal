"""
=============================================================================
AeroVision AI - Qdrant Vector DB Compatibility Bootstrap
=============================================================================
Ensures seamless in-memory and local Qdrant execution on Windows environments
where Windows Smart App Control restricts unsigned cygrpc binaries.
"""

import sys
import types
from unittest.mock import MagicMock

def bootstrap_qdrant_environment():
    """Patches missing or blocked grpc bindings for local in-memory Qdrant usage."""
    if 'grpc' in sys.modules and not isinstance(sys.modules['grpc'], MagicMock):
        return

    try:
        import grpc
    except Exception:
        class MockGrpc(types.ModuleType):
            UnaryUnaryClientInterceptor = type('U1', (), {})
            UnaryStreamClientInterceptor = type('U2', (), {})
            StreamUnaryClientInterceptor = type('U3', (), {})
            StreamStreamClientInterceptor = type('U4', (), {})
            ClientCallDetails = type('U5', (), {})
            Compression = MagicMock()
            ChannelCredentials = MagicMock()
            CallCredentials = MagicMock()
            AuthMetadataContext = MagicMock()
            AuthMetadataPlugin = MagicMock()
            AuthMetadataPluginCallback = MagicMock()
            ServerCredentials = MagicMock()
            Server = MagicMock()
            Channel = MagicMock()
            StatusCode = MagicMock()
            RpcError = Exception

            def __getattr__(self, name):
                return MagicMock()

        class MockGrpcAio(types.ModuleType):
            UnaryUnaryClientInterceptor = type('A1', (), {})
            UnaryStreamClientInterceptor = type('A2', (), {})
            StreamUnaryClientInterceptor = type('A3', (), {})
            StreamStreamClientInterceptor = type('A4', (), {})
            ClientCallDetails = type('A5', (), {})

            def __getattr__(self, name):
                return MagicMock()

        aio_mod = MockGrpcAio('grpc.aio')
        grpc_mod = MockGrpc('grpc')
        grpc_mod.aio = aio_mod

        sys.modules['grpc'] = grpc_mod
        sys.modules['grpc.aio'] = aio_mod

bootstrap_qdrant_environment()

# Suppress harmless shutdown teardown warning from QdrantClient.__del__
try:
    from qdrant_client import QdrantClient
    def safe_del(self):
        try:
            self.close()
        except Exception:
            pass
    QdrantClient.__del__ = safe_del
except Exception:
    pass
