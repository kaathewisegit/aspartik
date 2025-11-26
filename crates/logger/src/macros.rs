#[macro_export]
macro_rules! log {
	(
		target: $target:expr, $level:expr,
		$($key:ident = $value:expr),*
	) => {'scope: {
		use $crate::serde::{Serialize, ser::{SerializeStruct, Serializer}};
		use $crate::Level;

		#[allow(non_camel_case_types)]
		#[derive(Debug)]
		struct Tmp<$($key),*> {
			target: &'static str,
			level: Level,

			$($key: $key,)*
		}

		#[allow(non_camel_case_types)]
		impl<$($key),*> Serialize for Tmp<$($key),*>
		where
			$($key: Serialize),*
		{
			fn serialize<S>(
				&self,
				serializer: S,
			) -> Result<S::Ok, S::Error>
			where
				S: Serializer,
			{
				let mut state = serializer.serialize_struct("Tmp", 2 + $($crate::one(&self.$key) + )* 0)?;

				state.serialize_field("target", self.target)?;
				state.serialize_field("level", &self.level)?;

				$(
				state.serialize_field(
					stringify!($key),
					&self.$key,
				)?;
				)*

				state.end()
			}
		}

		let tmp = Tmp {
			target: $target,
			level: $level,

			$($key: &$value),*
		};

		let Some(logger) = $crate::LOGGER.get() else {
			break 'scope;
		};

		logger.log(&$level, &tmp);
	}};

	($level:expr, $($key:ident = $value:expr),*) => {
		$crate::log!(
			target: concat!(
				std::module_path!(),
				":", std::line!(), ":", std::column!(),
			),
			$level, $($key = $value),*
		);
	};
}

#[macro_export]
macro_rules! trace {
	(target: $target:expr, $($key:ident = $value:expr),*) => {
		$crate::log!(target: $target, $crate::Level::Trace, $($key = $value),*)
	};
	($($key:ident = $value:expr),*) => {
		$crate::log!($crate::Level::Trace, $($key = $value),*)
	};
}

#[macro_export]
macro_rules! debug {
	(target: $target:expr, $($key:ident = $value:expr),*) => {
		$crate::log!(target: $target, $crate::Level::Debug, $($key = $value),*)
	};
	($($key:ident = $value:expr),*) => {
		$crate::log!($crate::Level::Debug, $($key = $value),*)
	};
}

#[macro_export]
macro_rules! info {
	(target: $target:expr, $($key:ident = $value:expr),*) => {
		$crate::log!(target: $target, $crate::Level::Info, $($key = $value),*)
	};
	($($key:ident = $value:expr),*) => {
		$crate::log!($crate::Level::Info, $($key = $value),*)
	};
}

#[macro_export]
macro_rules! warn {
	(target: $target:expr, $($key:ident = $value:expr),*) => {
		$crate::log!(target: $target, $crate::Level::Warn, $($key = $value),*)
	};
	($($key:ident = $value:expr),*) => {
		$crate::log!($crate::Level::Warn, $($key = $value),*)
	};
}

#[macro_export]
macro_rules! error {
	(target: $target:expr, $($key:ident = $value:expr),*) => {
		$crate::log!(target: $target, $crate::Level::Error, $($key = $value),*)
	};
	($($key:ident = $value:expr),*) => {
		$crate::log!($crate::Level::Error, $($key = $value),*)
	};
}
