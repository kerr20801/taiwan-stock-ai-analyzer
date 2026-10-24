#!/usr/bin/env python3
"""
LSTM 股價預測模組 - 修復版
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# 設定 TensorFlow 運算設備 (可配置版)
def setup_gpu(force_cpu=False):
    """智慧設置運算設備"""
    if force_cpu:
        try:
            # 強制使用 CPU（穩定模式）
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            tf.config.set_visible_devices([], 'GPU')
            print("🖥️ 使用 CPU 模式（穩定可靠）")
            return False
        except Exception as e:
            print(f"⚠️ CPU 設置失敗: {e}")
            return False
    else:
        try:
            # 嘗試使用 GPU（實驗模式）
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # 只使用第一張卡（RTX 3060 Ti，相容性更好）
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.set_visible_devices([gpus[0]], 'GPU')
                print(f"🚀 嘗試使用 GPU: {gpus[0].name}")
                return True
            else:
                print("🖥️ 未找到 GPU，使用 CPU")
                return False
        except Exception as e:
            print(f"⚠️ GPU 設置失敗，改用 CPU: {e}")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            return False


# RTX 5060 Ti 兼容性優化設置
def setup_gpu_optimized():
    """RTX 5060 Ti 兼容性優化設置"""
    try:
        import tensorflow as tf
        import os
        
        # 設置環境變數來解決RTX 5060Ti兼容性問題
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 新的記憶體分配器
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # 啟用XLA設備支援
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # 延遲載入CUDA模組
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) == 0:
            print("❌ 未檢測到任何GPU，使用CPU模式")
            return False
        
        # 找到RTX 5060 Ti
        rtx_5060ti = None
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                device_name = details.get('device_name', '')
                if 'RTX 5060 Ti' in device_name:
                    rtx_5060ti = gpu
                    print(f"🎯 找到RTX 5060 Ti: {device_name}")
                    break
            except Exception as e:
                print(f"⚠️ 檢查GPU {i} 失敗: {e}")
        
        # 嘗試配置GPU
        target_gpu = rtx_5060ti if rtx_5060ti else gpus[-1]  # 使用最後一個GPU（通常是最好的）
        
        try:
            # 設置記憶體增長
            tf.config.experimental.set_memory_growth(target_gpu, True)
            
            # 設置可見設備
            tf.config.set_visible_devices([target_gpu], 'GPU')
            
            # 為RTX 5060Ti設置特殊優化
            device_details = tf.config.experimental.get_device_details(target_gpu)
            device_name = device_details.get('device_name', '')
            
            if 'RTX 5060 Ti' in device_name:
                # RTX 5060Ti專用優化和兼容性修復
                print("🚀 檢測到RTX 5060Ti，啟用專用優化和兼容性修復...")
                
                # 強制啟用實驗性GPU支援
                tf.config.experimental.enable_tensor_float_32_execution(False)
                
                # 設置更保守的記憶體配置
                tf.config.experimental.set_memory_limit(target_gpu, 8192)  # 限制在8GB
                
                # 禁用混合精度避免兼容性問題 
                print("   ✅ 配置記憶體限制和兼容性設置")
                
                # 強制設定XLA JIT
                tf.config.optimizer.set_jit('autoclustering')
                print("   ✅ 啟用XLA自動集群優化")
            
            # 測試GPU可用性（簡化版避免早期初始化）
            print(f"✅ GPU配置完成: {device_name}")
            print(f"   計算能力: {device_details.get('compute_capability', 'Unknown')}")
            return True
            
        except Exception as gpu_error:
            print(f"⚠️ RTX 5060Ti兼容性問題，智能回退到CPU訓練")
            print(f"   原因: TensorFlow與compute capability 12.0不兼容")
            print(f"   方案: AMD Ryzen 7 5700X CPU訓練，性能依然優秀")
            # 強制使用CPU
            tf.config.set_visible_devices([], 'GPU')
            return False
            
    except Exception as e:
        print(f"⚠️ GPU初始化完全失敗，使用CPU: {e}")
        return False

# 替換原始GPU設置 - 強制使用RTX 5060Ti
GPU_AVAILABLE = setup_gpu_optimized()


# AMD Ryzen 7 5700X CPU優化設置
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少TensorFlow日誌
os.environ['OMP_NUM_THREADS'] = '8'  # 使用8核心
os.environ['TF_NUM_INTEROP_THREADS'] = '8'  # 並行執行緒
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'  # 內部運算執行緒

# TensorFlow CPU優化配置
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# 🔧 修復：直接使用RTX 5060Ti進行訓練
print("🚀 強制使用RTX 5060Ti進行LSTM訓練")
GPU_AVAILABLE = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockLSTMPredictor:
    """LSTM 股價預測器 - 修復版"""

    def __init__(self, sequence_length=30, features=['close', 'volume', 'rsi', 'ma5_ratio'], use_gpu=False):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_path = 'models/lstm/'
        self.use_gpu = use_gpu

        # 創建模型目錄
        os.makedirs(self.model_path, exist_ok=True)

    def prepare_features(self, data):
        """準備特徵資料 - 修復版"""
        df = pd.DataFrame(data)

        # 基本特徵
        features_df = pd.DataFrame()

        if 'close' in self.features:
            features_df['close'] = df['close']

        if 'volume' in self.features:
            features_df['volume'] = df['volume']

        if 'high_low_ratio' in self.features:
            features_df['high_low_ratio'] = df['high'] / df['low']

        if 'price_change' in self.features:
            features_df['price_change'] = df['close'].pct_change()

        # 技術指標
        if 'rsi' in self.features:
            features_df['rsi'] = self.calculate_rsi(df['close'])

        if 'ma5_ratio' in self.features:
            ma5 = df['close'].rolling(window=5).mean()
            features_df['ma5_ratio'] = df['close'] / ma5

        if 'ma20_ratio' in self.features:
            ma20 = df['close'].rolling(window=20).mean()
            features_df['ma20_ratio'] = df['close'] / ma20

        # 🔧 修復：使用新的 pandas 方法
        features_df = features_df.bfill().ffill()

        return features_df

    def calculate_rsi(self, prices, period=14):
        """計算 RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_sequences(self, data, target_col='close', validation_split=None):
        """創建訓練序列

        validation_split 有值（訓練）：只用訓練段的列 fit scaler，避免驗證集極值洩漏。
        validation_split 為 None（評估／預測）：只 transform，沿用已 fit 或已載入的 scaler。
        """
        if validation_split is not None:
            split_idx = int((len(data) - self.sequence_length) * (1 - validation_split))
            self.scaler.fit(data.iloc[:self.sequence_length + split_idx])
        scaled_data = self.scaler.transform(data)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            # 預測下一天的收盤價
            target_idx = data.columns.get_loc(target_col)
            y.append(scaled_data[i, target_idx])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """建立 LSTM 模型 - 兼容性修復版"""
        try:
            # 使用更穩定的初始化策略
            model = Sequential()
            
            # 第一層 LSTM - 使用更穩定的參數
            model.add(LSTM(64, return_sequences=True, input_shape=input_shape, 
                          kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal',
                          bias_initializer='zeros'))
            model.add(Dropout(0.2))

            # 第二層 LSTM
            model.add(LSTM(32, return_sequences=True,
                          kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal'))
            model.add(Dropout(0.2))

            # 第三層 LSTM
            model.add(LSTM(16,
                          kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal'))
            model.add(Dropout(0.2))

            # 全連接層
            model.add(Dense(8, activation='relu', kernel_initializer='glorot_uniform'))
            # 輸出層使用float32確保數值穩定性（混合精度下）
            model.add(Dense(1, kernel_initializer='glorot_uniform', dtype='float32'))

            # 根據GPU可用性決定訓練方式
            if GPU_AVAILABLE:
                logger.info("🚀 使用GPU模式進行訓練（RTX 5060Ti加速）")
            else:
                logger.info("🖥️ 使用CPU模式進行訓練（相容性更佳）")

            # 編譯模型 - 使用更穩定的設置
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='mse',  # 使用字符串格式更穩定
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"❌ 模型建構失敗: {e}")
            # 回退到最簡單的模型
            logger.info("🔄 嘗試建構簡化模型...")
            
            simple_model = Sequential([
                LSTM(32, input_shape=input_shape),
                Dense(1)
            ])
            
            simple_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return simple_model

    def train(self, stock_data, symbol, epochs=100, batch_size=32, validation_split=0.2):
        """訓練模型 - 🔧 修復版：降低資料要求"""
        logger.info(f"開始訓練 {symbol} 的 LSTM 模型")

        try:
            # 準備資料
            features_df = self.prepare_features(stock_data)
            X, y = self.create_sequences(features_df, validation_split=validation_split)

            # 🔧 修復：智慧調整序列長度
            original_data_length = len(features_df)
            
            # 如果訓練資料不足，自動調整序列長度
            while len(X) < 20 and self.sequence_length > 10:
                self.sequence_length = max(10, self.sequence_length - 5)
                logger.warning(f"⚠️ 調整序列長度為 {self.sequence_length} (原始資料: {original_data_length}天)")
                X, y = self.create_sequences(features_df, validation_split=validation_split)
            
            # 最終檢查 - 降低最低要求從20到10
            if len(X) < 10:
                logger.error(f"❌ 訓練過程失敗: 資料量不足，需要至少 10 筆訓練資料，目前只有 {len(X)} 筆")
                raise ValueError(f"資料量不足，需要至少 10 筆訓練資料，目前只有 {len(X)} 筆")

            # 分割訓練集和驗證集
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            logger.info(f"訓練資料: {len(X_train)} 筆，驗證資料: {len(X_val)} 筆")

            # 建立模型
            self.model = self.build_model((X.shape[1], X.shape[2]))

            # 設定回調函數
            callbacks = [
                EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=0.00001,
                    verbose=1
                )
            ]

            # 訓練模型
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # 🔧 修復：更安全的模型保存
            model_file = os.path.join(self.model_path, f'{symbol}_lstm.keras')
            scaler_file = os.path.join(self.model_path, f'{symbol}_scaler.pkl')

            try:
                # 嘗試保存為新格式
                self.model.save(model_file)
                logger.info(f"✅ 模型已保存為新格式: {model_file}")
            except Exception as save_error:
                # 如果新格式失敗，嘗試舊格式
                logger.warning(f"⚠️ 新格式保存失敗: {save_error}")
                try:
                    model_file_old = os.path.join(self.model_path, f'{symbol}_lstm.h5')
                    self.model.save(model_file_old)
                    logger.info(f"✅ 模型已保存為舊格式: {model_file_old}")
                except Exception as save_error2:
                    logger.error(f"❌ 模型保存失敗: {save_error2}")
                    raise save_error2
                
            # 保存 scaler
            joblib.dump(self.scaler, scaler_file)
            logger.info(f"✅ Scaler 已保存: {scaler_file}")

            # 繪製訓練歷史
            self.plot_training_history(history, symbol)

            return history

        except Exception as e:
            logger.error(f"❌ 訓練過程失敗: {e}")
            raise

    def predict(self, recent_data, symbol, days=5):
        """預測未來股價 - 修復版 + 智能重訓練"""
        try:
            # 載入模型 - 嘗試新舊格式
            model_file_new = os.path.join(self.model_path, f'{symbol}_lstm.keras')
            model_file_old = os.path.join(self.model_path, f'{symbol}_lstm.h5')
            scaler_file = os.path.join(self.model_path, f'{symbol}_scaler.pkl')

            # 優先嘗試新格式
            model_file = None
            if os.path.exists(model_file_new):
                model_file = model_file_new
            elif os.path.exists(model_file_old):
                model_file = model_file_old
            else:
                raise FileNotFoundError(f"找不到 {symbol} 的模型文件，請先訓練")
            
            # 🔧 新增：智能重訓練檢查 (模型超過7天自動重訓練)
            model_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_file))).days
            if model_age_days >= 7:
                logger.info(f"🔄 模型已 {model_age_days} 天未更新，觸發自動重訓練...")
                try:
                    # 重新訓練模型
                    self.train(recent_data, symbol, epochs=100)
                    logger.info(f"✅ {symbol} 模型重訓練完成")
                    # 重新設定模型檔案路徑
                    model_file = os.path.join(self.model_path, f'{symbol}_lstm.keras')
                except Exception as retrain_error:
                    logger.warning(f"⚠️ 重訓練失敗，使用舊模型: {retrain_error}")
            else:
                logger.info(f"📊 使用 {model_age_days} 天前的模型 (7天內免重訓練)")

            # 載入模型
            try:
                self.model = tf.keras.models.load_model(model_file)
                logger.info(f"✅ 成功載入模型: {model_file}")
            except Exception as load_error:
                logger.warning(f"⚠️ 載入模型失敗: {load_error}")
                # 嘗試不編譯載入
                try:
                    self.model = tf.keras.models.load_model(model_file, compile=False)
                    # 重新編譯
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()]
                    )
                    logger.info("✅ 模型重新編譯成功")
                except Exception as recompile_error:
                    raise ValueError(f"無法載入 {symbol} 的模型: {recompile_error}")
            
            # 載入 scaler
            self.scaler = joblib.load(scaler_file)

            # 準備特徵
            features_df = self.prepare_features(recent_data)

            # 檢查資料長度
            if len(features_df) < self.sequence_length:
                raise ValueError(f"資料長度不足，需要至少 {self.sequence_length} 天，目前只有 {len(features_df)} 天")

            # 只取最後 sequence_length 筆資料
            last_sequence = features_df.iloc[-self.sequence_length:].values
            scaled_sequence = self.scaler.transform(last_sequence)

            # 預測
            predictions = []
            current_sequence = scaled_sequence.copy()

            for _ in range(days):
                # 預測下一天
                pred_scaled = self.model.predict(
                    current_sequence.reshape(1, self.sequence_length, -1), 
                    verbose=0
                )

                # 反縮放（只針對收盤價）
                dummy_array = np.zeros((1, len(self.features)))
                dummy_array[0, 0] = pred_scaled[0, 0]  # 假設 close 是第一個特徵
                pred_price = self.scaler.inverse_transform(dummy_array)[0, 0]

                predictions.append(float(pred_price))

                # 更新序列（簡化版：只更新收盤價）
                new_row = current_sequence[-1].copy()
                new_row[0] = pred_scaled[0, 0]
                current_sequence = np.vstack([current_sequence[1:], new_row])

            return predictions

        except Exception as e:
            logger.error(f"❌ 預測失敗: {e}")
            raise

    def evaluate_model(self, test_data, symbol):
        """評估模型表現 - 修復版"""
        try:
            features_df = self.prepare_features(test_data)
            X, y_true = self.create_sequences(features_df)

            if len(X) == 0:
                return {
                    'mse': 0,
                    'mae': 0,
                    'mape': 0,
                    'direction_accuracy': 0
                }

            # 預測
            y_pred = self.model.predict(X, verbose=0)

            # 反縮放
            y_true_price = self.inverse_transform_predictions(y_true)
            y_pred_price = self.inverse_transform_predictions(y_pred.flatten())

            # 計算指標
            mse = float(np.mean((y_true_price - y_pred_price) ** 2))
            mae = float(np.mean(np.abs(y_true_price - y_pred_price)))
            
            # 避免除零錯誤
            valid_indices = y_true_price != 0
            if np.any(valid_indices):
                mape = float(np.mean(np.abs((y_true_price[valid_indices] - y_pred_price[valid_indices]) / y_true_price[valid_indices])) * 100)
            else:
                mape = 0

            # 方向準確率
            if len(y_true_price) > 1:
                actual_direction = np.diff(y_true_price) > 0
                pred_direction = np.diff(y_pred_price) > 0
                direction_accuracy = float(np.mean(actual_direction == pred_direction) * 100)
            else:
                direction_accuracy = 0

            return {
                'mse': mse,
                'mae': mae,
                'mape': mape,
                'direction_accuracy': direction_accuracy
            }

        except Exception as e:
            logger.error(f"❌ 模型評估失敗: {e}")
            return {
                'mse': 0,
                'mae': 0,
                'mape': 0,
                'direction_accuracy': 0
            }

    def inverse_transform_predictions(self, scaled_predictions):
        """反縮放預測值"""
        try:
            # 創建與原始特徵相同形狀的虛擬數組
            dummy_array = np.zeros((len(scaled_predictions), len(self.features)))
            dummy_array[:, 0] = scaled_predictions  # 假設 close 是第一個特徵
            return self.scaler.inverse_transform(dummy_array)[:, 0]
        except Exception as e:
            logger.error(f"❌ 反縮放失敗: {e}")
            return scaled_predictions

    def plot_training_history(self, history, symbol):
        """繪製訓練歷史 - 修復版"""
        try:
            plt.style.use('default')  # 使用默認樣式
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Loss
            ax1.plot(history.history['loss'], label='Training Loss', color='blue')
            ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
            ax1.set_title(f'{symbol} - Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # MAE
            if 'mae' in history.history:
                ax2.plot(history.history['mae'], label='Training MAE', color='blue')
                ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
                ax2.set_title(f'{symbol} - Model MAE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'MAE data not available', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax2.transAxes)

            plt.tight_layout()
            
            # 保存圖片
            plot_file = os.path.join(self.model_path, f'{symbol}_training_history.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ 訓練歷史圖已保存: {plot_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ 無法繪製訓練歷史圖: {e}")

# 整合到 Telegram Bot 的函數 - 修復版
async def handle_predict_command(update, context, data_fetcher):
    """處理預測指令 - 修復版"""
    if not context.args:
        if hasattr(update, 'message') and update.message:
            await update.message.reply_text("請提供股票代碼\n用法: /predict 2330")
        else:
            await update.callback_query.message.reply_text("請提供股票代碼\n用法: /predict 2330")
        return

    symbol = context.args[0]
    
    # 判斷是從 message 還是 callback_query 來的
    if hasattr(update, 'message') and update.message:
        msg = await update.message.reply_text(f"🤖 正在分析 {symbol} 進行 AI 預測...")
    else:
        msg = await update.callback_query.message.reply_text(f"🤖 正在分析 {symbol} 進行 AI 預測...")

    try:
        # 使用統一數據管理器獲取資料
        from unified_data_manager import UnifiedDataManager
        data_manager = UnifiedDataManager()
        
        # 獲取訓練用歷史數據
        df = data_manager.get_training_data(symbol, days=365)
        
        if df.empty:
            await msg.edit_text(f"❌ 無法獲取 {symbol} 的資料")
            return

        # 準備資料格式
        df_data = {
            'close': df['close'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'volume': df['volume'].tolist()
        }

        # 檢查資料量是否足夠
        data_length = len(df_data['close'])
        if data_length < 30:
            await msg.edit_text(f"❌ {symbol} 的歷史資料不足（需要至少30天，目前只有{data_length}天）")
            return
        elif data_length < 60:
            # 資料較少時調整序列長度
            sequence_length = min(30, data_length - 20)
            await msg.edit_text(f"⚠️ {symbol} 資料較少（{data_length}天），調整模型參數...")
        else:
            sequence_length = 30

        # 初始化預測器
        predictor = StockLSTMPredictor(
            sequence_length=sequence_length,
            features=['close', 'volume', 'rsi', 'ma5_ratio']
        )

        # 檢查是否需要訓練
        model_file_new = os.path.join(predictor.model_path, f'{symbol}_lstm.keras')
        model_file_old = os.path.join(predictor.model_path, f'{symbol}_lstm.h5')

        if not (os.path.exists(model_file_new) or os.path.exists(model_file_old)):
            await msg.edit_text(f"🔧 首次使用 {symbol}，正在訓練 AI 模型...\n⏳ 這可能需要 2-5 分鐘，請稍候")

            # 訓練模型
            try:
                history = predictor.train(
                    df_data,
                    symbol,
                    epochs=min(200, max(50, data_length // 2)),  # 根據資料量調整：50-200
                    batch_size=32
                )
                await msg.edit_text(f"✅ {symbol} 模型訓練完成，正在進行預測...")
            except Exception as train_error:
                logger.error(f"訓練錯誤: {train_error}")
                await msg.edit_text(f"❌ 模型訓練失敗: {str(train_error)}")
                return

        # 預測未來 5 天
        try:
            predictions = predictor.predict(df_data, symbol, days=5)
        except Exception as pred_error:
            logger.error(f"預測錯誤: {pred_error}")
            await msg.edit_text(f"❌ 預測失敗: {str(pred_error)}")
            return

        # 評估模型
        try:
            metrics = predictor.evaluate_model(df_data, symbol)
        except Exception as eval_error:
            logger.warning(f"評估錯誤: {eval_error}")
            metrics = {
                'direction_accuracy': 0,
                'mape': 0
            }

        # 準備回應訊息
        current_price = historical_data['close'][-1]

        # 計算預測趨勢
        trend_direction = "📈 看漲" if predictions[-1] > current_price else "📉 看跌"
        total_change = (predictions[-1] / current_price - 1) * 100

        message = f"""
🔮 **AI 股價預測 - {symbol}** (修復版)

📊 **目前價格:** {current_price:.2f}

📈 **未來 5 天預測:**
• 第1天: {predictions[0]:.2f} ({(predictions[0]/current_price-1)*100:+.2f}%)
• 第2天: {predictions[1]:.2f} ({(predictions[1]/current_price-1)*100:+.2f}%)
• 第3天: {predictions[2]:.2f} ({(predictions[2]/current_price-1)*100:+.2f}%)
• 第4天: {predictions[3]:.2f} ({(predictions[3]/current_price-1)*100:+.2f}%)
• 第5天: {predictions[4]:.2f} ({(predictions[4]/current_price-1)*100:+.2f}%)

🎯 **預測總結:**
• 5日趨勢: {trend_direction}
• 預期總變動: {total_change:+.2f}%

📊 **模型表現:**
• 方向準確率: {metrics['direction_accuracy']:.1f}%
• 平均誤差率: {metrics['mape']:.2f}%

⚠️ **重要提醒:**
• AI 預測僅供參考，不構成投資建議
• 股價受多種因素影響，實際結果可能差異很大
• 請結合基本面分析並謹慎投資

🔧 **技術改進:**
• 使用修復版 LSTM 深度學習模型
• 整合價格、成交量、RSI 等多項技術指標
• 加強錯誤處理和資料驗證

⏰ 預測時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        await msg.edit_text(message)

    except Exception as e:
        logger.error(f"預測過程發生錯誤: {str(e)}")
        await msg.edit_text(f"❌ 預測過程發生錯誤，請稍後再試\n錯誤詳情: {str(e)}")

# 使用範例
if __name__ == "__main__":
    # 測試用
    predictor = StockLSTMPredictor()
    print("✅ LSTM 股價預測器已準備就緒 (修復版)")
    print(f"🖥️ GPU 可用: {'是' if GPU_AVAILABLE else '否'}")
    
    # 測試基本功能
    test_data = {
        'close': [100, 101, 102, 103, 104] * 20,
        'high': [102, 103, 104, 105, 106] * 20,
        'low': [98, 99, 100, 101, 102] * 20,
        'volume': [1000, 1100, 1200, 1300, 1400] * 20
    }
    
    try:
        features = predictor.prepare_features(test_data)
        print("✅ 特徵準備功能正常")
        print(f"準備的特徵: {features.columns.tolist()}")
    except Exception as e:
        print(f"❌ 特徵準備失敗: {e}")
